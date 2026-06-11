"""Compute the water density profile as a function of distance from the TiO2 surface.

Reads an xyz trajectory, identifies water O atoms by coordination (O bonded to Ti
belongs to the lattice; the rest are water/solute), and histograms their z-distance
from the nearest surface.  Detects solvation-layer boundaries from local density
minima and writes them to a text file for use in BEC_part1.py.

Example:
    python rho.py --traj md_out.xyz --nframes 500
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread, read
from ase.neighborlist import NeighborList
from scipy.signal import argrelextrema


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute water density profile along z for TiO2-water MD."
    )
    parser.add_argument("--traj", default="md_out.xyz",
                        help="Path to the xyz trajectory file.")
    parser.add_argument("--nframes", type=int, default=500,
                        help="Number of final frames to analyze.")
    parser.add_argument("--dz-max", type=float, default=12.0,
                        help="Maximum z-distance from surface to histogram (Å).")
    parser.add_argument("--bin-width", type=float, default=0.05,
                        help="Histogram bin width (Å).")
    parser.add_argument("--bond-cutoff", type=float, default=2.4,
                        help="Ti-O bond cutoff radius for lattice O identification (Å).")
    parser.add_argument("--output-plot", default="water_density_profile.png",
                        help="Output density plot filename.")
    parser.add_argument("--output-data", default="density_profile.txt",
                        help="Output density data filename.")
    parser.add_argument("--output-cuts", default="detected_layer_cuts.txt",
                        help="Output file for detected layer boundary positions.")
    return parser.parse_args()


def identify_atom_roles(atoms, bond_cutoff):
    """Return indices of Ti/lattice-O (slab) and water O atoms."""
    symbols = atoms.get_chemical_symbols()
    cutoffs = [bond_cutoff / 2.0] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)

    ti_indices, lattice_o_indices, water_o_indices = [], [], []
    for i, sym in enumerate(symbols):
        if sym == 'Ti':
            ti_indices.append(i)
        elif sym == 'O':
            neighbors, _ = nl.get_neighbors(i)
            n_ti = sum(1 for idx in neighbors if symbols[idx] == 'Ti')
            if n_ti >= 2:
                lattice_o_indices.append(i)
            else:
                water_o_indices.append(i)

    slab_indices = np.concatenate([ti_indices, lattice_o_indices])
    return np.array(ti_indices), np.array(lattice_o_indices), np.array(water_o_indices), slab_indices


def main():
    args = parse_args()

    if not os.path.exists(args.traj):
        raise FileNotFoundError(f"Trajectory not found: {args.traj}")

    # Count total frames to determine start index
    with open(args.traj) as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    lines_per_frame = natoms + 2
    total_frames = len(lines) // lines_per_frame
    start_idx = max(0, total_frames - args.nframes)
    n_analyze = total_frames - start_idx
    print(f"Total frames: {total_frames}. Analyzing last {n_analyze} frames.")

    first_atoms = read(args.traj, index=start_idx)
    _, _, water_o_indices, slab_indices = identify_atom_roles(first_atoms, args.bond_cutoff)
    print(f"Water O atoms: {len(water_o_indices)}, slab atoms: {len(slab_indices)}")

    cell = first_atoms.get_cell()
    lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]

    dz_min = 0.0
    n_bins = int((args.dz_max - dz_min) / args.bin_width)
    z_grid = np.linspace(dz_min, args.dz_max, n_bins, endpoint=False) + args.bin_width / 2
    accumulated_hist = np.zeros(n_bins)

    print("Processing frames...")
    frame_count = 0
    for atoms in iread(args.traj, index=slice(start_idx, total_frames)):
        pos = atoms.positions

        if len(slab_indices) > 0:
            slab_z = pos[slab_indices, 2]
            z_mean = np.mean(slab_z)
            slab_lower = slab_z[slab_z < z_mean]
            slab_upper = slab_z[slab_z >= z_mean]
            tol = 0.8
            z_surf_bottom = (np.mean(slab_lower[slab_lower > np.max(slab_lower) - tol])
                             if len(slab_lower) > 0 else 0.0)
            z_surf_top = (np.mean(slab_upper[slab_upper < np.min(slab_upper) + tol])
                          if len(slab_upper) > 0 else lz)
        else:
            z_surf_bottom, z_surf_top = 0.0, lz

        if len(water_o_indices) > 0:
            z_water = pos[water_o_indices, 2]
            mid = (z_surf_bottom + z_surf_top) / 2.0
            delta_z = np.where(z_water <= mid,
                               z_water - z_surf_bottom,
                               z_surf_top - z_water)
            hist, _ = np.histogram(delta_z, bins=n_bins, range=(dz_min, args.dz_max))
            accumulated_hist += hist

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  {frame_count}/{n_analyze} frames")

    # Convert to density (g/mL)
    vol_factor = 2 * (lx * ly * args.bin_width) * 1e-24
    density = (accumulated_hist / n_analyze / vol_factor) * (18.015 / 6.022e23)

    # Detect solvation-layer boundaries from density minima
    order = int(0.5 / args.bin_width)
    minima_indices = argrelextrema(density, np.less, order=order)[0]
    minima_z = z_grid[minima_indices]
    valid_minima = minima_z[(minima_z > 1.2) & (minima_z < 8.0)]
    detected_cuts = list(np.round(valid_minima[:3], 3))
    print(f"Detected layer boundaries: {detected_cuts}")

    np.savetxt(args.output_data, np.c_[z_grid, density],
               header='Delta_Z(A) Density(g/mL)', fmt='%.6f')
    with open(args.output_cuts, 'w') as f:
        f.write(f"LAYER_CUTS = {detected_cuts}\n")

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(z_grid, density, color='black', linewidth=2)
    for cut in detected_cuts:
        plt.axvline(x=cut, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.8)
    plt.xlim(1.0, 10.0)
    plt.xlabel(r'$\Delta z$ [Å]')
    plt.ylabel('Density [g/mL]')
    plt.title('Water density profile near TiO2 surface')
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
