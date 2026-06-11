"""Compute H3O+--water oxygen radial distribution functions from MD trajectories.

Scans all run* subdirectories under BASE_DIR for .traj files, analyzes the
second half of each trajectory, and writes per-directory CSV files.
"""

import gc
import glob
import os

import numpy as np
import pandas as pd
from ase.geometry import get_distances
from ase.io import Trajectory
from ase.neighborlist import neighbor_list
from tqdm import tqdm


BASE_DIR = "./"
R_MAX = 10.0
N_BINS = 200
OUTPUT_FILENAME = "H3O_Water_RDF.csv"
FRAME_STRIDE = 100  # analyze one frame per N steps


def get_species_oxygens(atoms, r_cut_h3o=1.35, expected_h3o_count=4):
    """Return oxygen indices for H3O+ cores and water molecules.

    Identifies H3O+ by selecting the `expected_h3o_count` oxygens with the
    highest H coordination number.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    o_indices = np.where(symbols == "O")[0]

    i_idx, j_idx = neighbor_list("ij", atoms, r_cut_h3o)
    o_h_count = {o_id: 0 for o_id in o_indices}
    for idx_i, idx_j in zip(i_idx, j_idx):
        if symbols[idx_i] == "O" and symbols[idx_j] == "H":
            o_h_count[idx_i] += 1

    sorted_o = sorted(o_h_count.keys(), key=lambda x: o_h_count[x], reverse=True)
    h3o_o = sorted_o[:expected_h3o_count]
    water_o = sorted_o[expected_h3o_count:]
    return np.array(h3o_o, dtype=int), np.array(water_o, dtype=int)


def compute_rdf_histogram(dist_matrix, bins, r_max):
    hist, _ = np.histogram(dist_matrix, bins=bins, range=(0, r_max))
    return hist


if __name__ == "__main__":
    target_dirs = sorted(
        d for d in glob.glob(os.path.join(BASE_DIR, "run*")) if os.path.isdir(d)
    )
    print(f"Found {len(target_dirs)} directories in {BASE_DIR}")

    for subdir in target_dirs:
        dir_name = os.path.basename(subdir)
        print(f"\n{'=' * 50}\nProcessing: {dir_name}")

        traj_files = glob.glob(os.path.join(subdir, "*.traj"))
        if not traj_files:
            print(f"  [Skipped] No .traj file in {dir_name}")
            continue

        traj_path = traj_files[0]
        out_path = os.path.join(subdir, OUTPUT_FILENAME)
        print(f"  Trajectory: {traj_path}\n  Output:     {out_path}")

        try:
            full_traj = Trajectory(traj_path, "r")
            total_frames = len(full_traj)
            start_frame = int(total_frames * 0.5)  # use second half
            traj_slice = full_traj[start_frame::FRAME_STRIDE]

            print(f"  Total frames: {total_frames}")
            print(f"  Analyzing frames {start_frame}–{total_frames} with stride {FRAME_STRIDE}")
            print(f"  Frames to process: {len(traj_slice)}")

            if len(traj_slice) == 0:
                print("  [Warning] No frames selected.")
                continue

            dr = R_MAX / N_BINS
            r_values = np.linspace(dr / 2, R_MAX - dr / 2, N_BINS)
            total_histogram = np.zeros(N_BINS, dtype=np.float64)
            accumulated_density = 0.0
            total_h3o_count = 0
            frames_processed = 0

            for i, atoms in tqdm(enumerate(traj_slice), total=len(traj_slice),
                                 desc=dir_name, leave=False):
                h3o_o, water_o = get_species_oxygens(atoms, r_cut_h3o=1.30, expected_h3o_count=4)
                if len(h3o_o) == 0 or len(water_o) == 0:
                    continue

                _, dist_matrix = get_distances(
                    atoms.positions[h3o_o], atoms.positions[water_o],
                    cell=atoms.cell, pbc=atoms.pbc
                )
                total_histogram += compute_rdf_histogram(dist_matrix.flatten(), N_BINS, R_MAX)

                rho_frame = len(water_o) / atoms.get_volume()
                accumulated_density += rho_frame
                total_h3o_count += len(h3o_o)
                frames_processed += 1

                if (i + 1) % 100 == 0:
                    gc.collect()

            if frames_processed > 0:
                avg_rho = accumulated_density / frames_processed
                shell_volumes = 4.0 * np.pi * (r_values ** 2) * dr
                g_r = total_histogram / (total_h3o_count * shell_volumes * avg_rho)
                pd.DataFrame({"r_angstrom": r_values, "g_r": g_r}).to_csv(out_path, index=False)
                print(f"  [Success] Saved to {out_path}")
            else:
                print("  [Warning] No valid frames processed.")

            del full_traj, traj_slice
            gc.collect()

        except Exception as e:
            import traceback
            print(f"  [Error] {subdir}: {e}")
            traceback.print_exc()

    print("\nAll tasks completed.")
