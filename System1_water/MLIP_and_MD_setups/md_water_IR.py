"""Run water MD simulations for a directory of trained MLIP models.

Example:
    python md_water_IR.py \
        --init-config path/to/md_out.xyz \
        --model-dir path/to/models \
        --output-root path/to/md_runs
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run water MD simulations for trained MLIP models."
    )
    parser.add_argument(
        "--init-config",
        required=True,
        type=Path,
        help="Initial structure file. The frame selected by --init-index is used.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing model subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("md_runs"),
        type=Path,
        help="Directory where MD output folders will be written.",
    )
    parser.add_argument(
        "--model-dir-pattern",
        default="mlp-train-*",
        help="Glob pattern for model subdirectories under --model-dir.",
    )
    parser.add_argument(
        "--model-file-pattern",
        default="*_stagetwo.model",
        help="Glob pattern for model files inside each model subdirectory.",
    )
    parser.add_argument(
        "--system-name",
        default="H2O",
        help="Prefix for trajectory and log output files.",
    )
    parser.add_argument(
        "--init-index",
        default=-1,
        type=int,
        help="Frame index read from --init-config.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used by MACECalculator, for example cuda or cpu.",
    )
    parser.add_argument(
        "--default-dtype",
        default="float64",
        choices=("float32", "float64"),
        help="Default floating-point dtype used by MACECalculator.",
    )
    parser.add_argument("--temperature", default=300.0, type=float, help="Temperature in K.")
    parser.add_argument("--timestep", default=0.25, type=float, help="MD timestep in fs.")
    parser.add_argument(
        "--equilibration-steps",
        default=1000,
        type=int,
        help="Number of equilibration MD steps.",
    )
    parser.add_argument(
        "--nsteps",
        default=200000,
        type=int,
        help="Number of production MD steps.",
    )
    parser.add_argument(
        "--fmax",
        default=0.03,
        type=float,
        help="BFGS force convergence threshold.",
    )
    parser.add_argument(
        "--traj-interval",
        default=1,
        type=int,
        help="Interval for writing the ASE trajectory.",
    )
    parser.add_argument(
        "--log-interval",
        default=500,
        type=int,
        help="Interval for writing the standard MD log and extra diagnostics.",
    )
    parser.add_argument(
        "--xyz-interval",
        default=100,
        type=int,
        help="Interval for appending frames to md_out.xyz.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.init_config.is_file():
        raise FileNotFoundError(f"Initial configuration not found: {args.init_config}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"Model directory not found: {args.model_dir}")
    for name in (
        "equilibration_steps",
        "nsteps",
        "traj_interval",
        "log_interval",
        "xyz_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    if args.timestep <= 0:
        raise ValueError("--timestep must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.fmax <= 0:
        raise ValueError("--fmax must be positive.")


def find_model_files(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    model_entries = []
    model_dirs = sorted(
        path
        for path in args.model_dir.glob(args.model_dir_pattern)
        if path.is_dir() and path.name != "checkpoints"
    )

    for model_dir in model_dirs:
        model_files = sorted(
            path
            for path in model_dir.glob(args.model_file_pattern)
            if path.is_file() and not path.name.startswith("._")
        )
        if not model_files:
            print(f"Skipping {model_dir}: no {args.model_file_pattern} file found.")
            continue
        if len(model_files) > 1:
            print(f"Using first matching model in {model_dir}: {model_files[0].name}")
        model_entries.append((model_dir, model_files[0]))

    return model_entries


def run_name_from_model_dir(model_dir: Path) -> str:
    prefix = "mlp-train-"
    name = model_dir.name
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return f"run_{name}"


def log_extra_info(dyn, atoms, log_path: Path) -> None:
    """Append total force and center-of-mass velocity diagnostics."""
    total_force = np.sum(atoms.get_forces(), axis=0)
    velocities = atoms.get_velocities()
    masses = atoms.get_masses().reshape(-1, 1)
    com_velocity = np.sum(velocities * masses, axis=0) / np.sum(masses)

    with log_path.open("a") as handle:
        handle.write(
            f"{dyn.nsteps} "
            f"{total_force[0]:.6f} {total_force[1]:.6f} {total_force[2]:.6f} "
            f"{com_velocity[0]:.6f} {com_velocity[1]:.6f} {com_velocity[2]:.6f}\n"
        )


def run_md_for_model(model_path: Path, output_dir: Path, args: argparse.Namespace) -> bool:
    """Run geometry optimization, equilibration, and production MD for one model."""
    from ase import units
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
    from ase.md.logger import MDLogger
    from ase.md.npt import NPT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import BFGS
    from mace.calculators import MACECalculator

    output_dir.mkdir(parents=True, exist_ok=True)
    abs_model_path = model_path.resolve()

    print(f"\n{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {abs_model_path}")
    print(f"Initial configuration: {args.init_config}")
    print(f"{'=' * 60}")

    try:
        calculator = MACECalculator(
            model_paths=str(abs_model_path),
            device=args.device,
            default_dtype=args.default_dtype,
        )

        atoms = read(str(args.init_config), index=args.init_index).copy()
        atoms.calc = calculator

        print("Running geometry optimization with BFGS...")
        optimizer = BFGS(atoms, logfile=str(output_dir / "optimization.log"))
        optimizer.run(fmax=args.fmax)

        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

        dyn = NPT(
            atoms,
            args.timestep * units.fs,
            temperature_K=args.temperature,
            ttime=10 * units.fs,
            pfactor=None,
            externalstress=0.0,
        )

        trajectory = Trajectory(str(output_dir / f"{args.system_name}.traj"), "w", atoms)
        dyn.attach(trajectory.write, interval=args.traj_interval)

        md_logger = MDLogger(
            dyn,
            atoms,
            logfile=str(output_dir / f"{args.system_name}.log"),
            header=True,
            stress=False,
            mode="w",
        )
        dyn.attach(md_logger, interval=args.log_interval)

        xyz_path = output_dir / "md_out.xyz"
        dyn.attach(
            lambda: write(str(xyz_path), atoms, format="extxyz", append=True),
            interval=args.xyz_interval,
        )

        extra_log = output_dir / "md_extra.log"
        dyn.attach(log_extra_info, args.log_interval, dyn, atoms, extra_log)

        print(f"Running equilibration for {args.equilibration_steps} steps...")
        dyn.run(args.equilibration_steps)

        print(f"Running production for {args.nsteps} steps...")
        dyn.run(args.nsteps)
        print(f"Completed: {output_dir}")
        return True

    except Exception as exc:
        print(f"Run failed for {model_path}: {exc}")
        return False


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    model_entries = find_model_files(args)
    if not model_entries:
        raise SystemExit(f"No model files found in {args.model_dir}.")

    completed = 0
    for model_dir, model_file in model_entries:
        output_dir = args.output_root / run_name_from_model_dir(model_dir)
        completed += int(run_md_for_model(model_file, output_dir, args))

    print(
        f"\nBatch completed: {completed}/{len(model_entries)} "
        "model runs finished successfully."
    )


if __name__ == "__main__":
    main()
