import os
import argparse
import numpy as np
import torch
from torch import cuda
from mace.calculators import MACECalculator
from ase.io import read, write
from ase import units
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase.md.logger import MDLogger
from ase.io.trajectory import Trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Run NPT MD simulation with a MACE model.")
    parser.add_argument("--model_path", required=True, type=str,
                        help="Path to the trained MACE stagetwo model.")
    parser.add_argument("--init_config", required=True, type=str,
                        help="Path to the initial configuration xyz file (last frame is used).")
    parser.add_argument("--system_name", type=str, default="2MHCl",
                        help="System name prefix for output files.")
    parser.add_argument("--output_root", type=str, default="MDrun",
                        help="Root directory for MD output.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K.")
    parser.add_argument("--timestep", type=float, default=0.25, help="Timestep in fs.")
    parser.add_argument("--nsteps", type=int, default=400000, help="Production MD steps.")
    parser.add_argument("--equil_steps", type=int, default=10000, help="Equilibration steps.")
    return parser.parse_args()


DEVICE = 'cuda' if cuda.is_available() else 'cpu'


def log_extra_info(dyn, atoms, log_path):
    """Log total force and center-of-mass velocity at each interval."""
    total_force = np.sum(atoms.get_forces(), axis=0)
    velocities = atoms.get_velocities()
    masses = atoms.get_masses().reshape(-1, 1)
    com_velocity = np.sum(velocities * masses, axis=0) / np.sum(masses)
    with open(log_path, "a") as f:
        f.write(f"{dyn.nsteps} {total_force[0]:.6f} {total_force[1]:.6f} {total_force[2]:.6f} "
                f"{com_velocity[0]:.6f} {com_velocity[1]:.6f} {com_velocity[2]:.6f}\n")


def run_md(model_path, init_config_path, system_name, output_path, temperature, timestep, nsteps, equil_steps):
    """Run the full MD workflow for one model."""
    abs_model_path = os.path.abspath(model_path)
    script_root_dir = os.getcwd()

    print(f"\n{'='*60}", flush=True)
    print(f">>> Output:       {output_path}", flush=True)
    print(f">>> Model:        {abs_model_path}", flush=True)
    print(f">>> Init config:  {init_config_path}", flush=True)
    print(f"{'='*60}", flush=True)

    os.makedirs(output_path, exist_ok=True)
    os.chdir(output_path)

    try:
        calculator = MACECalculator(
            model_paths=abs_model_path,
            device=DEVICE,
            default_dtype="float64"
        )

        atoms = read(init_config_path, index=-1).copy()
        atoms.calc = calculator

        print("--- Structure optimization (BFGS) ---", flush=True)
        optimizer = BFGS(atoms, logfile='optimization.log')
        optimizer.run(fmax=0.03)

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

        dyn = NPT(atoms,
                  timestep * units.fs,
                  temperature_K=temperature,
                  ttime=25 * units.fs,
                  pfactor=None,  # NVT
                  externalstress=0.0)

        traj = Trajectory(f'{system_name}.traj', 'w', atoms)
        dyn.attach(traj.write, interval=1)
        dyn.attach(MDLogger(dyn, atoms, logfile=f'{system_name}.log',
                            header=True, stress=False, mode='w'), interval=500)
        dyn.attach(lambda: write('md_out.xyz', atoms, format='extxyz', append=True), interval=100)
        dyn.attach(log_extra_info, 500, dyn, atoms, "md_extra.log")

        print(f"--- Equilibration ({equil_steps} steps) ---", flush=True)
        dyn.run(equil_steps)

        print(f"--- Production ({nsteps} steps) ---", flush=True)
        dyn.run(nsteps)
        print(f">>> Completed: {output_path}", flush=True)

    except Exception as e:
        print(f"!!! Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(script_root_dir)


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: model file not found: {args.model_path}", flush=True)
    else:
        model_dir = os.path.basename(os.path.dirname(os.path.abspath(args.model_path)))
        run_name = "run_" + model_dir.replace("train-MLP", "")
        output_path = os.path.join(args.output_root, run_name)
        run_md(args.model_path, args.init_config, args.system_name, output_path,
               args.temperature, args.timestep, args.nsteps, args.equil_steps)

    print("\nMD simulation completed.", flush=True)
