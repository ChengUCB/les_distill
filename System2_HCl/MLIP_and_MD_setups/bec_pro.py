import gc
import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm
from ase.io import Trajectory
from ase.neighborlist import neighbor_list

import mace
from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose BEC-derived current into H3O+, solvation shell, and bulk water."
    )
    parser.add_argument("--model_path", required=True, type=str,
                        help="Path to the trained MACE stagetwo model.")
    parser.add_argument("--traj_path", type=str, default="2MHCl.traj",
                        help="Path to the ASE trajectory file.")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame index.")
    parser.add_argument("--end", type=int, default=400000,
                        help="End frame index.")
    parser.add_argument("--checkpoint_interval", type=int, default=50000,
                        help="Save a checkpoint every N frames.")
    return parser.parse_args()


def get_species_indices_strict(atoms, r_cut_h3o=1.35, r_cut_shell=3.0, expected_h3o_count=4):
    """Identify H3O+ cores and solvation shells; remainder is bulk water."""
    symbols = np.array(atoms.get_chemical_symbols())
    all_indices = np.arange(len(atoms))

    i_idx, j_idx, d_vals = neighbor_list('ijd', atoms, r_cut_h3o)

    o_indices = np.where(symbols == 'O')[0]
    o_h_count = {o_id: 0 for o_id in o_indices}
    o_grouping = {o_id: [] for o_id in o_indices}

    h_ownership = {}
    for k, (idx_i, idx_j) in enumerate(zip(i_idx, j_idx)):
        if symbols[idx_i] == 'H' and symbols[idx_j] == 'O':
            dist = d_vals[k]
            if idx_i not in h_ownership or dist < h_ownership[idx_i][1]:
                h_ownership[idx_i] = (idx_j, dist)

    for h_id, (o_id, _) in h_ownership.items():
        o_grouping[o_id].append(h_id)
        o_h_count[o_id] += 1

    sorted_o = sorted(o_h_count.keys(), key=lambda x: o_h_count[x], reverse=True)
    h3o_central_o_list = sorted_o[:expected_h3o_count]

    h3o_core_indices = []
    for o_id in h3o_central_o_list:
        h3o_core_indices.append(o_id)
        h3o_core_indices.extend(o_grouping[o_id])
    h3o_core_idx = np.array(h3o_core_indices, dtype=int)

    if len(h3o_central_o_list) > 0:
        i_shell, j_shell = neighbor_list('ij', atoms, r_cut_shell)
        mask = np.isin(i_shell, h3o_central_o_list)
        shell_atoms = j_shell[mask]
        h3o_complex_idx = np.unique(np.concatenate([h3o_core_idx, shell_atoms])).astype(int)
    else:
        h3o_complex_idx = np.array([], dtype=int)

    water_indices = np.setdiff1d(all_indices, h3o_complex_idx)
    return h3o_core_idx, h3o_complex_idx, water_indices


if __name__ == "__main__":
    args = parse_args()

    CHECKPOINT_PREFIX = f'HCl_checkpoint_{args.start}_{args.end}'
    FINAL_FILENAME = f'HCl_decomp_results_{args.start}_{args.end}.pkl'
    SYSTEM_TOTAL_CHARGE = 0.0
    BEC_FACTOR = 1.0

    print(f"  Start frame:         {args.start}")
    print(f"  End frame:           {args.end}")
    print(f"  Trajectory:          {args.traj_path}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print("Loading MACE model...")

    DEVICE = torch_tools.init_device('cuda')
    model = torch.load(args.model_path, map_location=DEVICE)
    model = model.to(DEVICE)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = model.r_max

    full_traj = Trajectory(args.traj_path, 'r')
    real_end = min(args.end, len(full_traj))
    traj_slice = full_traj[args.start:real_end]
    print(f"Processing frames {args.start} to {real_end} ({len(traj_slice)} frames)")

    results = {'J_h3o': [], 'J_complex': [], 'J_bulk_water': [], 'J_total': []}

    for i, atoms in tqdm(enumerate(traj_slice), total=len(traj_slice)):
        global_frame_idx = args.start + i

        atoms.info['total_charge'] = SYSTEM_TOTAL_CHARGE
        config = data.config_from_atoms(atoms)
        batch_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=float(r_max))
        batch = torch_geometric.Batch.from_data_list([batch_data]).to(DEVICE)
        batch_dict = batch.to_dict()

        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                batch_dict[k] = v.to(torch.float64)

        output = model(batch_dict, compute_stress=False, compute_bec=True)
        BEC = output['BEC'] * BEC_FACTOR
        velocity = torch.tensor(atoms.get_velocities(), dtype=BEC.dtype, device=DEVICE)
        dP_atomic = torch.bmm(BEC, velocity.unsqueeze(-1)).squeeze(-1)

        h3o_core, h3o_complex, water_idx = get_species_indices_strict(atoms, expected_h3o_count=4)

        J_total = torch.sum(dP_atomic, dim=0)
        J_h3o = torch.sum(dP_atomic[h3o_core], dim=0) if len(h3o_core) > 0 else torch.zeros(3, device=DEVICE, dtype=torch.float64)
        J_complex = torch.sum(dP_atomic[h3o_complex], dim=0) if len(h3o_complex) > 0 else torch.zeros(3, device=DEVICE, dtype=torch.float64)
        J_bulk = torch.sum(dP_atomic[water_idx], dim=0) if len(water_idx) > 0 else torch.zeros(3, device=DEVICE, dtype=torch.float64)

        results['J_h3o'].append(J_h3o.detach().cpu().numpy())
        results['J_complex'].append(J_complex.detach().cpu().numpy())
        results['J_bulk_water'].append(J_bulk.detach().cpu().numpy())
        results['J_total'].append(J_total.detach().cpu().numpy())

        del output, BEC, velocity, dP_atomic, batch, batch_dict, config
        if (i + 1) % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        current_count = i + 1
        if current_count % args.checkpoint_interval == 0:
            ckpt_name = f'{CHECKPOINT_PREFIX}_{global_frame_idx + 1}.pkl'
            print(f"Saving checkpoint to {ckpt_name}...")
            with open(ckpt_name, 'wb') as f:
                pickle.dump({k: np.array(v) for k, v in results.items()}, f)

    print(f"Saving final results to {FINAL_FILENAME}...")
    with open(FINAL_FILENAME, 'wb') as f:
        pickle.dump({k: np.array(v) for k, v in results.items()}, f)
    print("Done.")
