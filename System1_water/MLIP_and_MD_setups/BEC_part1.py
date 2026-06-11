"""Compute the total BEC-derived current for a trajectory segment.

Example:
    python BEC_part1.py \
        --model-path path/to/model_stagetwo.model \
        --traj-path H2O.traj \
        --start-frame 0 \
        --end-frame 2000
"""

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate total BEC-derived current for a trajectory segment."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to the trained MACE stagetwo model.",
    )
    parser.add_argument(
        "--traj-path",
        default=Path("H2O.traj"),
        type=Path,
        help="Path to the ASE trajectory file.",
    )
    parser.add_argument(
        "--start-frame",
        required=True,
        type=int,
        help="Start frame index, inclusive.",
    )
    parser.add_argument(
        "--end-frame",
        required=True,
        type=int,
        help="End frame index, exclusive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output pickle file. Defaults to bec_results_<start>_<end>.pkl.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, for example cuda or cpu.",
    )
    parser.add_argument(
        "--bec-factor",
        default=1.0,
        type=float,
        help="Multiplicative factor applied to the predicted BEC tensor.",
    )
    parser.add_argument(
        "--system-total-charge",
        default=0.0,
        type=float,
        help="Total charge stored in atoms.info before model evaluation.",
    )
    parser.add_argument(
        "--empty-cache-interval",
        default=500,
        type=int,
        help="Clear CUDA cache and run garbage collection every N frames.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {args.traj_path}")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative.")
    if args.end_frame <= args.start_frame:
        raise ValueError("--end-frame must be greater than --start-frame.")
    if args.empty_cache_interval < 1:
        raise ValueError("--empty-cache-interval must be at least 1.")


def load_model(model_path: Path, device: torch.device, mace_utils):
    print(f"Loading MACE model from {model_path}...")
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    z_table = mace_utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", None)
    print("MACE model loaded successfully.")
    return model, z_table, float(model.r_max), heads


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return Path(f"bec_results_{args.start_frame}_{args.end_frame}.pkl")


def calculate_segment(args: argparse.Namespace) -> None:
    from ase.io import Trajectory
    from mace import data
    from mace.tools import torch_geometric, torch_tools, utils
    from tqdm import tqdm

    validate_inputs(args)
    device = torch_tools.init_device(args.device)
    model, z_table, r_max, heads = load_model(args.model_path, device, utils)

    print(f"Reading trajectory from {args.traj_path}...")
    trajectory = Trajectory(str(args.traj_path), "r")

    total_frames = len(trajectory)
    start_frame = args.start_frame
    end_frame = min(args.end_frame, total_frames)

    if start_frame >= total_frames:
        raise ValueError(
            f"--start-frame {start_frame} is outside the trajectory "
            f"with {total_frames} frames."
        )
    if end_frame != args.end_frame:
        print(
            f"Warning: --end-frame {args.end_frame} exceeds the trajectory length. "
            f"Using {total_frames} instead."
        )
    if end_frame <= start_frame:
        raise ValueError("The selected trajectory segment is empty.")

    print(f"Processing frames [{start_frame}, {end_frame})...")
    trajectory_segment = trajectory[start_frame:end_frame]
    total_dp = []

    for frame_offset, atoms in tqdm(
        enumerate(trajectory_segment), total=len(trajectory_segment)
    ):
        atoms.info["total_charge"] = args.system_total_charge

        config = data.config_from_atoms(atoms)
        atomic_data = data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
            heads=heads,
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(device)
        batch_dict = batch.to_dict()

        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                batch_dict[key] = value.to(torch.float64)

        output = model(batch_dict, compute_stress=False, compute_bec=True)
        bec = output["BEC"] * args.bec_factor
        velocity = torch.as_tensor(atoms.get_velocities(), dtype=bec.dtype, device=device)

        atomic_dp = torch.bmm(bec, velocity.unsqueeze(-1)).squeeze(-1)
        total_dp.append(torch.sum(atomic_dp, dim=0).detach().cpu().numpy())

        del config, atomic_data, data_loader, batch, batch_dict
        del output, bec, velocity, atomic_dp

        if (frame_offset + 1) % args.empty_cache_interval == 0:
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    results = np.array(total_dp)
    output_file = output_path(args)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(results)} frames to {output_file}...")
    with output_file.open("wb") as handle:
        pickle.dump({"total_dp": results}, handle)

    print("Segment processing completed.")


if __name__ == "__main__":
    calculate_segment(parse_args())
