"""Geometry optimization of the TiO2-NaCl-water structure with a CACE model.

Three-phase protocol:
  1. FIRE global optimization (removes large forces quickly)
  2. Short Langevin MD at low temperature (cooperative surface-water relaxation)
  3. FIRE final refinement

Example:
    python opt2.py \\
        --input tio2_nacl.xyz \\
        --model-path best_model.pth
"""

import argparse

import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE, LBFGS
from cace.calculators import CACECalculator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-phase geometry optimization for TiO2-water with CACE."
    )
    parser.add_argument("--input", default="tio2_nacl.xyz",
                        help="Path to the input structure (last frame is used).")
    parser.add_argument("--model-path", default="best_model.pth",
                        help="Path to the CACE model (.pth).")
    parser.add_argument("--temperature", type=float, default=330.0,
                        help="Temperature (K) for the low-T MD relaxation phase.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (cuda or cpu).")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading CACE model from {args.model_path}...")
    cace_nnp = torch.load(args.model_path, map_location=args.device)
    calculator = CACECalculator(
        model_path=cace_nnp,
        device=args.device,
        energy_key='CACE_energy',
        forces_key='CACE_forces',
        compute_stress=False,
    )

    print(f"Reading structure from {args.input}...")
    atoms = read(args.input, index=-1)
    atoms.set_constraint()  # remove all FixAtoms constraints for global optimization
    atoms.calc = calculator

    # Use slightly heavier H mass to allow a larger timestep
    masses = atoms.get_masses()
    for i in range(len(atoms)):
        if atoms[i].symbol == 'H':
            masses[i] = 10.0
    atoms.set_masses(masses)

    forces = atoms.get_forces()
    fmax_init = np.max(np.linalg.norm(forces, axis=1))
    print(f"Initial max force: {fmax_init:.4f} eV/Å")

    # Phase 1: FIRE rough optimization
    print("\n=== Phase 1: FIRE rough optimization ===")
    opt_fire = FIRE(atoms, maxstep=0.05, dt=0.05)
    opt_fire.run(fmax=1.0, steps=300)

    opt_lbfgs = LBFGS(atoms)
    opt_lbfgs.run(fmax=0.1, steps=500)
    write('phase1_fire.xyz', atoms)

    # Phase 2: short low-temperature MD for cooperative relaxation
    print("\n=== Phase 2: Low-temperature MD relaxation ===")
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    dyn = Langevin(atoms,
                   timestep=0.5 * units.fs,
                   temperature_K=args.temperature,
                   friction=0.02)
    logger = MDLogger(dyn, atoms, '-', header=True, mode="w")
    dyn.attach(logger, interval=10)
    dyn.run(1000)
    write('phase2_md.xyz', atoms)

    # Phase 3: FIRE final refinement
    print("\n=== Phase 3: FIRE final refinement ===")
    opt_final = FIRE(atoms, maxstep=0.05, dt=0.05)
    opt_final.run(fmax=0.1, steps=1500)

    output_file = 'optimized_final.xyz'
    write(output_file, atoms)
    print(f"\nOptimization complete. Structure saved to {output_file}")


if __name__ == "__main__":
    main()
