"""Run NVT MD simulation of the TiO2-water interface with a CACE model.

Example:
    python cace_md_NVT.py \\
        --model-path best_model_DFT_SCAN.pth \\
        --init-config ini.xyz \\
        --temperature 330 \\
        --nsteps 200000
"""

import argparse
import time

import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.logger import MDLogger
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import cace
from cace.calculators import CACECalculator
import torch._functorch.config
import torch._inductor.config

torch._functorch.config.donated_buffer = False
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')


def parse_args():
    parser = argparse.ArgumentParser(description="Run NVT MD with a CACE model.")
    parser.add_argument("--model-path", default="best_model_DFT_SCAN.pth", type=str,
                        help="Path to the trained CACE model (.pth).")
    parser.add_argument("--init-config", default="ini.xyz", type=str,
                        help="Path to the initial configuration xyz file.")
    parser.add_argument("--temperature", type=float, default=330.0,
                        help="Temperature in K.")
    parser.add_argument("--timestep", type=float, default=0.3,
                        help="MD timestep in fs.")
    parser.add_argument("--equil-steps", type=int, default=50000,
                        help="Equilibration steps (trajectory not recorded).")
    parser.add_argument("--nsteps", type=int, default=200000,
                        help="Production MD steps.")
    parser.add_argument("--traj-file", default="md_TiO2-water.traj", type=str,
                        help="Output trajectory file.")
    parser.add_argument("--log-file", default="md_TiO2-water.log", type=str,
                        help="MD log file.")
    parser.add_argument("--xyz-output", default="md_out.xyz", type=str,
                        help="XYZ output file (every 500 steps).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (cuda or cpu).")
    return parser.parse_args()


def optimize_model_hybrid(model, atoms):
    """Bypass torch.compile for Ewald modules, which have dynamic grid sizes."""
    for module in model.modules():
        name = module.__class__.__name__
        if 'Ewald' in name:
            module.forward = torch._dynamo.disable(module.forward)
            if hasattr(module, 'dl'):
                box_lengths = atoms.cell.lengths()
                module.static_Nk = [max(1, int(l / module.dl) + 1) for l in box_lengths]
            print(f"  Ewald bypass applied: {name}")


def log_extra_info(atoms, dyn):
    total_force = np.sum(atoms.get_forces(), axis=0)
    velocities = atoms.get_velocities()
    m = atoms.get_masses().reshape(-1, 1)
    com_velocity = np.sum(velocities * m, axis=0) / np.sum(m)
    with open("md_extra.log", "a") as f:
        f.write(f"{dyn.get_number_of_steps()} "
                f"{total_force[0]:.6f} {total_force[1]:.6f} "
                f"{com_velocity[0]:.6e}\n")


def main():
    args = parse_args()
    DEVICE = args.device

    cace_nnp = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    temp_atoms = read(args.init_config, index=-1)

    optimize_model_hybrid(cace_nnp, temp_atoms)
    print("Compiling model with torch.compile...")
    cace_nnp = torch.compile(cace_nnp, mode='reduce-overhead')

    atoms = temp_atoms.copy()
    symbols = np.array(atoms.get_chemical_symbols())
    masses = atoms.get_masses()
    masses[symbols == 'H'] = 1.00794
    masses[symbols == 'O'] = 15.9994
    atoms.set_masses(masses)
    print(f"System size: {len(atoms)} atoms")

    for key in list(atoms.info.keys()):
        if key in ('energy', 'stress', 'free_energy', 'virial') \
                or 'energy' in key or 'stress' in key:
            del atoms.info[key]
    if 'forces' in atoms.arrays:
        del atoms.arrays['forces']

    calculator = CACECalculator(
        model_path=cace_nnp,
        device=DEVICE,
        compute_stress=False,
        energy_key='CACE_energy',
        forces_key='CACE_forces',
    )
    atoms.calc = calculator

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    dyn = NPT(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        ttime=100 * units.fs,
        pfactor=None,  # NVT (no barostat)
        externalstress=0.0,
    )

    print(f"Equilibration ({args.equil_steps} steps, trajectory not recorded)...")
    dyn.run(args.equil_steps)

    traj = Trajectory(args.traj_file, 'w', atoms)
    dyn.attach(traj.write, interval=1)

    md_logger = MDLogger(dyn, atoms, logfile=args.log_file, header=True, stress=False, mode='w')
    dyn.attach(md_logger, interval=100)

    dyn.attach(lambda: write(args.xyz_output, atoms, format='extxyz', append=True), interval=500)
    dyn.attach(log_extra_info, interval=100, atoms=atoms, dyn=dyn)

    print(f"Production MD: {args.nsteps} steps at {args.temperature} K...")
    start_time = time.time()
    dyn.run(args.nsteps)
    elapsed = time.time() - start_time

    print("-" * 50)
    print(f"MD completed. Total time: {elapsed:.2f} s  "
          f"({elapsed / args.nsteps * 1000:.2f} ms/step)")
    print("-" * 50)


if __name__ == "__main__":
    main()
