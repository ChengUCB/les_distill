"""Train a CACE model with long-range electrostatics (Ewald) for the TiO2-water system.

Training data should be in extxyz format with 'energy' and 'forces' keys.
Atomic reference energies (E0s) are hard-coded from a prior linear regression;
to recompute them run calc_e0.py on your training set.

Example:
    python CACELES-train.py
"""

import logging
import sys

import torch
import torch.nn as nn
import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask
from cace.tools import Metrics


def log_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = log_exception
torch.set_default_dtype(torch.float32)
cace.tools.setup_logger(level='INFO')

CUTOFF = 5.5
TRAIN_PATH = './train-TiO2-water-DFT.xyz'
# Atomic reference energies (eV) from linear regression on training set
ATOMIC_ENERGIES = {
    1: 243.1915051693859,
    8: -953.7867084172028,
    22: -537.691230500944,
}


def build_model(device):
    radial_basis = BesselRBF(cutoff=CUTOFF, n_rbf=6, trainable=True)
    cutoff_fn = PolynomialCutoff(cutoff=CUTOFF)

    cace_representation = Cace(
        zs=[1, 8, 22],
        n_atom_basis=3,
        embed_receiver_nodes=True,
        cutoff=CUTOFF,
        cutoff_fn=cutoff_fn,
        radial_basis=radial_basis,
        n_radial_basis=12,
        max_l=3,
        max_nu=3,
        num_message_passing=0,
        type_message_passing=['Bchi'],
        args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
        forward_features=['atomic_numbers'],
        device=device,
        timeit=False,
    )
    cace_representation.to(device)

    sr_energy = cace.modules.atomwise.Atomwise(
        n_layers=3,
        output_key='SR_energy',
        n_hidden=[32, 16],
        use_batchnorm=False,
        add_linear_nn=True,
    )

    q = cace.modules.Atomwise(
        n_layers=3,
        n_hidden=[24, 12],
        n_out=1,
        per_atom_output_key='q',
        output_key='tot_q',
        residual=False,
        add_linear_nn=True,
        bias=False,
    )

    ep = cace.modules.EwaldPotential(
        dl=2,
        sigma=1.0,
        feature_key='q',
        output_key='ewald_potential',
        remove_self_interaction=False,
        aggregation_mode='sum',
    )

    e_add = cace.modules.FeatureAdd(
        feature_keys=['SR_energy', 'ewald_potential'],
        output_key='CACE_energy',
    )

    forces = cace.modules.Forces(
        energy_key='CACE_energy',
        forces_key='CACE_forces',
    )

    cace_nnp = NeuralNetworkPotential(
        input_modules=None,
        representation=cace_representation,
        output_modules=[sr_energy, q, ep, e_add, forces],
    )
    cace_nnp.to(device)
    return cace_nnp


def make_losses(energy_weight):
    energy_loss = cace.tasks.GetLoss(
        target_name='energy',
        predict_name='CACE_energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=energy_weight,
    )
    force_loss = cace.tasks.GetLoss(
        target_name='forces',
        predict_name='CACE_forces',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1000,
    )
    return [energy_loss, force_loss]


def main():
    logging.info("Loading training data")

    collection = cace.tasks.get_dataset_from_xyz(
        train_path=TRAIN_PATH,
        valid_fraction=0.1,
        seed=1,
        cutoff=CUTOFF,
        data_key={'energy': 'energy', 'forces': 'forces'},
        atomic_energies=ATOMIC_ENERGIES,
    )
    logging.info(f"Dataset size: {len(collection)}")

    batch_size = 8
    train_loader = cace.tasks.load_data_loader(collection=collection,
                                               data_type='train', batch_size=batch_size)
    valid_loader = cace.tasks.load_data_loader(collection=collection,
                                               data_type='valid', batch_size=8)

    device = cace.tools.init_device('cuda')
    logging.info(f"Device: {device}")

    cace_nnp = build_model(device)

    e_metric = Metrics(target_name='energy', predict_name='CACE_energy',
                       name='e/atom', per_atom=True)
    f_metric = Metrics(target_name='forces', predict_name='CACE_forces', name='f')

    optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}
    scheduler_args = {'step_size': 20, 'gamma': 0.5}

    # Warmup: small energy weight to stabilise force training first
    logging.info("Warmup training (5 x 40 epochs, energy_weight=0.1)")
    for i in range(5):
        task = TrainingTask(
            model=cace_nnp,
            losses=make_losses(energy_weight=0.1),
            metrics=[e_metric, f_metric],
            device=device,
            optimizer_args=optimizer_args,
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_args=scheduler_args,
            max_grad_norm=10,
            ema=True,
            ema_start=10,
            warmup_steps=5,
        )
        task.fit(train_loader, valid_loader, epochs=40, screen_nan=False)

    task.save_model('TiO2-NaCl-water-model-0.pth')
    cace_nnp.to(device)

    # Progressive energy weight schedule
    for stage, (energy_weight, epochs) in enumerate(
        [(1, 50), (10, 100), (100, 100), (1000, 100)], start=1
    ):
        logging.info(f"Stage {stage}: energy_weight={energy_weight}, epochs={epochs}")
        task.update_loss(make_losses(energy_weight=energy_weight))
        task.fit(train_loader, valid_loader, epochs=epochs, screen_nan=False, val_stride=10)
        task.save_model(f'TiO2-NaCl-water-model-{stage}.pth')
        cace_nnp.to(device)

    n_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {n_params}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.error("Fatal error in main loop", exc_info=True)
        sys.exit(1)
