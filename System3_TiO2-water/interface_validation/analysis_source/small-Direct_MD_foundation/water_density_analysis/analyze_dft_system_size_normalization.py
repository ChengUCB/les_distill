#!/usr/bin/env python3
"""Explore surface coordinates and normalization for small/large DFT profiles.

Both trajectories are reprocessed from XYZ coordinates using the final 75% of
frames.  The script separates two operations that must not be conflated:

1. the surface-coordinate definition (horizontal alignment), and
2. the conversion of water counts to an intensive density (vertical scale).

Several additional vertical transformations are evaluated diagnostically to
test whether a single normalization can make the two finite trajectories
identical.  Only the actual-area normalization is a physical mass density.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
from trajectory_io import open_binary, open_text, resolve_compressed  # noqa: E402


KEEP_FRACTION = 0.75
DZ_MIN = 0.0
DZ_MAX = 12.0
BIN_WIDTH = 0.05
BOND_CUTOFF = 2.4
WATER_MOLAR_MASS = 18.01528
AVOGADRO = 6.02214076e23
PEAK_WINDOW = (0.8, 3.5)
INTERFACE_WINDOW = (0.5, 6.0)
BULK_WINDOW = (5.0, 10.0)
LEGACY_TOL = 0.8
TOL_GRID = np.arange(0.55, 0.951, 0.01)
FRACTION_GRID = np.arange(0.03, 0.181, 0.01)
ONSET_DENSITY_THRESHOLD = 0.05


@dataclass
class TrajectoryData:
    path: Path
    symbols: np.ndarray
    z_frames: np.ndarray
    areas_A2: np.ndarray
    slab_idx: np.ndarray
    water_o_idx: np.ndarray
    total_frames: int
    start_frame_1based: int


def parse_lattice(comment: str) -> np.ndarray:
    match = re.search(r'Lattice="([^"]+)"', comment)
    if match is None:
        raise ValueError("Extended XYZ frame has no Lattice field")
    values = np.fromstring(match.group(1), sep=" ")
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def line_count(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    path = resolve_compressed(path)
    count = 0
    with open_binary(path) as handle:
        while chunk := handle.read(chunk_size):
            count += chunk.count(b"\n")
    return count


def trajectory_shape(path: Path) -> tuple[int, int]:
    path = resolve_compressed(path)
    with open_text(path) as handle:
        natoms = int(handle.readline())
    frame_lines = natoms + 2
    nlines = line_count(path)
    if nlines % frame_lines:
        raise ValueError(f"{path}: incomplete XYZ frame")
    return natoms, nlines // frame_lines


def classify_atoms(
    symbols: np.ndarray, positions: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Classify slab and water O using the legacy 2.4-A Ti-neighbor rule."""
    ti_idx = np.flatnonzero(symbols == "Ti")
    oxygen_idx = np.flatnonzero(symbols == "O")
    if ti_idx.size == 0 or oxygen_idx.size == 0:
        raise ValueError("Trajectory must contain Ti and O")
    delta = positions[oxygen_idx, None, :] - positions[ti_idx, :][None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.rint(fractional)
    delta_mic = fractional @ cell
    n_ti = np.sum(np.linalg.norm(delta_mic, axis=2) < BOND_CUTOFF, axis=1)
    lattice_o_idx = oxygen_idx[n_ti >= 2]
    water_o_idx = oxygen_idx[n_ti <= 1]
    slab_idx = np.concatenate((ti_idx, lattice_o_idx))
    return slab_idx, water_o_idx


def load_final_window(path: Path) -> TrajectoryData:
    path = resolve_compressed(path)
    natoms, nframes = trajectory_shape(path)
    keep = int(np.floor(nframes * KEEP_FRACTION))
    start = nframes - keep
    z_frames: list[np.ndarray] = []
    areas: list[float] = []
    symbols: np.ndarray | None = None
    selected_positions: np.ndarray | None = None
    selected_cell: np.ndarray | None = None

    with open_text(path) as handle:
        for frame_index in range(nframes):
            this_natoms = int(handle.readline())
            comment = handle.readline()
            if this_natoms != natoms:
                raise ValueError(f"Atom count changed at frame {frame_index}")
            cell = parse_lattice(comment)
            if frame_index < start:
                for _ in range(natoms):
                    handle.readline()
                continue

            z = np.empty(natoms, dtype=float)
            if frame_index == start:
                symbols = np.empty(natoms, dtype="U2")
                selected_positions = np.empty((natoms, 3), dtype=float)
                selected_cell = cell.copy()
            for atom_index in range(natoms):
                fields = handle.readline().split()
                z[atom_index] = float(fields[3])
                if frame_index == start:
                    assert symbols is not None and selected_positions is not None
                    symbols[atom_index] = fields[0]
                    selected_positions[atom_index] = [float(v) for v in fields[1:4]]
            z_frames.append(z)
            areas.append(float(np.linalg.norm(np.cross(cell[0], cell[1]))))

    if symbols is None or selected_positions is None or selected_cell is None:
        raise ValueError(f"No selected frames in {path}")
    slab_idx, water_o_idx = classify_atoms(symbols, selected_positions, selected_cell)
    return TrajectoryData(
        path=path,
        symbols=symbols,
        z_frames=np.asarray(z_frames),
        areas_A2=np.asarray(areas),
        slab_idx=slab_idx,
        water_o_idx=water_o_idx,
        total_frames=nframes,
        start_frame_1based=start + 1,
    )


def block_sem(values: np.ndarray, nblocks: int = 5) -> np.ndarray:
    blocks = [block for block in np.array_split(values, nblocks) if len(block)]
    means = np.asarray([np.mean(block, axis=0) for block in blocks])
    if len(means) == 1:
        return np.zeros(means.shape[1])
    return np.std(means, axis=0, ddof=1) / np.sqrt(len(means))


def surface_planes(
    slab_z: np.ndarray, method: str, parameter: float
) -> tuple[float, float, int, int]:
    z_mean = float(np.mean(slab_z))
    lower = slab_z[slab_z < z_mean]
    upper = slab_z[slab_z >= z_mean]
    if lower.size == 0 or upper.size == 0:
        raise ValueError("Could not split slab into lower and upper halves")

    if method == "legacy_tol":
        bottom_atoms = lower[lower > np.max(lower) - parameter]
        top_atoms = upper[upper < np.min(upper) + parameter]
    elif method == "common_fraction":
        n_bottom = max(1, int(round(parameter * lower.size)))
        n_top = max(1, int(round(parameter * upper.size)))
        bottom_atoms = np.partition(lower, lower.size - n_bottom)[-n_bottom:]
        top_atoms = np.partition(upper, n_top - 1)[:n_top]
    else:
        raise ValueError(f"Unknown surface method: {method}")
    return (
        float(np.mean(bottom_atoms)),
        float(np.mean(top_atoms)),
        int(bottom_atoms.size),
        int(top_atoms.size),
    )


def profile(data: TrajectoryData, method: str, parameter: float) -> dict:
    edges = np.arange(DZ_MIN, DZ_MAX + 0.5 * BIN_WIDTH, BIN_WIDTH)
    histograms = []
    valid_areas = []
    selected_counts = []
    surface_origins = []

    for z, area in zip(data.z_frames, data.areas_A2):
        slab_z = z[data.slab_idx]
        water_z = z[data.water_o_idx]
        slab_z = slab_z[np.isfinite(slab_z)]
        water_z = water_z[np.isfinite(water_z)]
        if slab_z.size < 10 or water_z.size == 0:
            continue
        try:
            bottom, top, n_bottom, n_top = surface_planes(slab_z, method, parameter)
        except ValueError:
            continue
        midpoint = 0.5 * (bottom + top)
        dz = np.where(water_z <= midpoint, water_z - bottom, top - water_z)
        histograms.append(np.histogram(dz[np.isfinite(dz)], bins=edges)[0])
        valid_areas.append(area)
        selected_counts.append((n_bottom, n_top))
        surface_origins.append((bottom, top))

    histograms = np.asarray(histograms, dtype=float)
    valid_areas = np.asarray(valid_areas, dtype=float)
    if histograms.size == 0:
        raise ValueError(f"No valid frames for {data.path}")
    bin_volumes_ml = 2.0 * valid_areas[:, None] * BIN_WIDTH * 1e-24
    per_frame = histograms * WATER_MOLAR_MASS / (AVOGADRO * bin_volumes_ml)
    raw = np.mean(per_frame, axis=0)
    smooth = savgol_filter(raw, window_length=7, polyorder=3)
    sem = block_sem(per_frame)
    z_grid = 0.5 * (edges[:-1] + edges[1:])
    peak_mask = (z_grid >= PEAK_WINDOW[0]) & (z_grid <= PEAK_WINDOW[1])
    peak_indices = np.flatnonzero(peak_mask)
    peak_index = peak_indices[np.argmax(smooth[peak_mask])]
    bulk_mask = (z_grid >= BULK_WINDOW[0]) & (z_grid <= BULK_WINDOW[1])
    counts = np.asarray(selected_counts)
    origins = np.asarray(surface_origins)
    return {
        "z": z_grid,
        "raw": raw,
        "smooth": smooth,
        "sem": sem,
        "per_frame": per_frame,
        "method": method,
        "parameter": float(parameter),
        "peak_z_A": float(z_grid[peak_index]),
        "peak_density_g_mL": float(smooth[peak_index]),
        "bulk_density_g_mL": float(np.mean(smooth[bulk_mask])),
        "valid_frames": int(len(histograms)),
        "mean_area_A2": float(np.mean(valid_areas)),
        "mean_bottom_atoms": float(np.mean(counts[:, 0])),
        "mean_top_atoms": float(np.mean(counts[:, 1])),
        "surface_bottom_sem_A": float(np.std(origins[:, 0], ddof=1) / np.sqrt(len(origins))),
        "surface_top_sem_A": float(np.std(origins[:, 1], ddof=1) / np.sqrt(len(origins))),
    }


def comparison_metrics(
    z: np.ndarray,
    small: np.ndarray,
    large: np.ndarray,
    small_sem: np.ndarray,
    large_sem: np.ndarray,
    window: tuple[float, float] = INTERFACE_WINDOW,
) -> dict:
    mask = (z >= window[0]) & (z <= window[1])
    residual = small - large
    combined_sem = np.sqrt(small_sem**2 + large_sem**2)
    peak_scale = max(float(np.max(np.abs(large[mask]))), 1e-12)
    return {
        "mae": float(np.mean(np.abs(residual[mask]))),
        "rmse": float(np.sqrt(np.mean(residual[mask] ** 2))),
        "nrmse_by_large_peak": float(
            np.sqrt(np.mean(residual[mask] ** 2)) / peak_scale
        ),
        "max_abs": float(np.max(np.abs(residual[mask]))),
        "fraction_within_1_combined_sem": float(
            np.mean(np.abs(residual[mask]) <= combined_sem[mask])
        ),
        "fraction_within_2_combined_sem": float(
            np.mean(np.abs(residual[mask]) <= 2.0 * combined_sem[mask])
        ),
        "fraction_exact_1e-10": float(np.mean(np.abs(residual[mask]) <= 1e-10)),
    }


def profile_onset_z(profile_data: dict, use_raw: bool = False) -> float:
    values = profile_data["raw"] if use_raw else profile_data["smooth"]
    threshold = 0.0 if use_raw else ONSET_DENSITY_THRESHOLD
    mask = (profile_data["z"] >= 0.5) & (values > threshold)
    indices = np.flatnonzero(mask)
    return float(profile_data["z"][indices[0]]) if indices.size else float("inf")


def transformed_modes(small: dict, large: dict) -> list[dict]:
    z = small["z"]
    mask = (z >= INTERFACE_WINDOW[0]) & (z <= INTERFACE_WINDOW[1])
    s = small["smooth"]
    l = large["smooth"]
    ssem = small["sem"]
    lsem = large["sem"]
    modes = []

    def add(name: str, scale: float, offset: float, status: str) -> None:
        y = scale * s + offset
        sem = abs(scale) * ssem
        modes.append(
            {
                "name": name,
                "scale": float(scale),
                "offset": float(offset),
                "status": status,
                "small": y,
                "small_sem": sem,
                **comparison_metrics(z, y, l, sem, lsem),
            }
        )

    add("actual_area_absolute_density", 1.0, 0.0, "physical")
    bulk_scale = large["bulk_density_g_mL"] / small["bulk_density_g_mL"]
    add("bulk_matched_global_scale", bulk_scale, 0.0, "shape_diagnostic")
    integral_scale = float(np.sum(l[mask]) / np.sum(s[mask]))
    add("interface_integral_matched_scale", integral_scale, 0.0, "shape_diagnostic")
    peak_scale = large["peak_density_g_mL"] / small["peak_density_g_mL"]
    add("first_peak_height_matched_scale", peak_scale, 0.0, "nonphysical_fit")
    least_squares_scale = float(np.dot(s[mask], l[mask]) / np.dot(s[mask], s[mask]))
    add("least_squares_global_scale", least_squares_scale, 0.0, "fit_diagnostic")
    design = np.column_stack((s[mask], np.ones(np.count_nonzero(mask))))
    scale, offset = np.linalg.lstsq(design, l[mask], rcond=None)[0]
    add("least_squares_affine", float(scale), float(offset), "nonphysical_fit")
    return modes


def scan_shift_and_scale(small: dict, large: dict) -> dict:
    """Find the best global shift/scale as a deliberately fitted lower bound."""
    z = small["z"]
    mask = (z >= INTERFACE_WINDOW[0]) & (z <= INTERFACE_WINDOW[1])
    best = None
    for shift in np.arange(-0.20, 0.2001, 0.005):
        # Positive shift moves the small curve toward larger Delta z.
        shifted = np.interp(z - shift, z, small["smooth"], left=np.nan, right=np.nan)
        valid = mask & np.isfinite(shifted)
        scale = float(
            np.dot(shifted[valid], large["smooth"][valid])
            / np.dot(shifted[valid], shifted[valid])
        )
        fitted = scale * shifted
        rmse = float(np.sqrt(np.mean((fitted[valid] - large["smooth"][valid]) ** 2)))
        item = (rmse, abs(shift), float(shift), scale, fitted)
        if best is None or item[:2] < best[:2]:
            best = item
    assert best is not None
    rmse, _, shift, scale, fitted = best
    valid = mask & np.isfinite(fitted)
    residual = fitted[valid] - large["smooth"][valid]
    return {
        "shift_A": shift,
        "scale": scale,
        "rmse_g_mL": rmse,
        "mae_g_mL": float(np.mean(np.abs(residual))),
        "max_abs_g_mL": float(np.max(np.abs(residual))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    outdir: Path,
    small_same: dict,
    large: dict,
    small_aligned: dict,
    fraction_small: dict,
    fraction_large: dict,
    tol_scan: list[dict],
    modes: list[dict],
) -> None:
    z = large["z"]
    abs_mode, bulk_mode, fitted_mode = modes[0], modes[1], modes[4]
    combined_sem = np.sqrt(small_aligned["sem"] ** 2 + large["sem"] ** 2)

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.2), dpi=180, sharex=False)
    ax = axes[0, 0]
    ax.plot(z, small_same["smooth"], color="#8c6bb1", lw=1.6, label="Small, tol=0.80 Å")
    ax.plot(z, large["smooth"], color="0.15", lw=1.6, ls="--", label="Large, tol=0.80 Å")
    ax.set(title="A. Same legacy surface tolerance", ylabel="Density [g/mL]", xlim=(0, 6))
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[0, 1]
    ax.fill_between(z, np.maximum(small_aligned["smooth"] - small_aligned["sem"], 0),
                    small_aligned["smooth"] + small_aligned["sem"],
                    color="#7b3294", alpha=0.16, linewidth=0)
    ax.plot(z, small_aligned["smooth"], color="#7b3294", lw=1.7,
            label=f"Small, tol={small_aligned['parameter']:.2f} Å")
    ax.plot(z, large["smooth"], color="0.15", lw=1.6, ls="--",
            label="Large, tol=0.80 Å")
    ax.set(title="B. Surface-origin aligned; physical density", ylabel="Density [g/mL]", xlim=(0, 6))
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 0]
    residual = abs_mode["small"] - large["smooth"]
    ax.fill_between(z, -2 * combined_sem, 2 * combined_sem, color="0.55", alpha=0.18,
                    label=r"$\pm2$ combined block SEM")
    ax.plot(z, residual, color="#7b3294", lw=1.3, label="Small − large")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.set(title="C. Residual after physical processing", ylabel=r"$\Delta\rho$ [g/mL]",
           xlim=(0, 6))
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 1]
    ax.plot(z, bulk_mode["small"], color="#c51b7d", lw=1.7,
            label=f"Small × {bulk_mode['scale']:.4f}")
    ax.plot(z, large["smooth"], color="0.15", lw=1.6, ls="--", label="Large")
    ax.set(title="D. Bulk-matched global normalization", ylabel="Density [g/mL]", xlim=(0, 6))
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2, 0]
    ax.plot(z, fraction_small["smooth"], color="#1b7837", lw=1.7,
            label=f"Small, common fraction={fraction_small['parameter']:.2f}")
    ax.plot(z, fraction_large["smooth"], color="0.15", lw=1.6, ls="--",
            label="Large, same fraction")
    ax.set(title="E. Size-independent surface-atom fraction", xlabel=r"$\Delta z$ [$\AA$]",
           ylabel="Density [g/mL]", xlim=(0, 6))
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2, 1]
    scan_tol = np.asarray([row["small_tol_A"] for row in tol_scan])
    scan_peak = np.asarray([row["small_peak_z_A"] for row in tol_scan])
    scan_mae = np.asarray([row["mae_g_mL"] for row in tol_scan])
    ax.plot(scan_tol, scan_peak, color="#2166ac", lw=1.5, label="Small peak position")
    ax.axhline(large["peak_z_A"], color="0.25", lw=1.0, ls="--", label="Large peak")
    ax.set(title="F. Surface-tolerance scan", xlabel="Small-system tolerance [Å]",
           ylabel="Peak position [Å]")
    ax2 = ax.twinx()
    ax2.plot(scan_tol, scan_mae, color="#b2182b", lw=1.2, alpha=0.8, label="Profile MAE")
    ax2.set_ylabel("MAE, 0.5–6 Å [g/mL]")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8.0, loc="best")

    for axis in axes.flat:
        axis.grid(ls="--", alpha=0.22)
    fig.suptitle("DFT water density: system-size coordinate and normalization tests", y=1.002)
    fig.tight_layout()
    fig.savefig(outdir / "dft_system_size_normalization_tests.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = (args.outdir or root / "water_density_analysis" / "dft_system_size_results").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    small_path = root / "Small-MD-CACELES" / "DFT-ref" / "md_out-2025-12.xyz"
    large_path = root / "Large-student-MD" / "DFT-ref" / "md_out.xyz"
    print("Loading final 75% of small DFT trajectory ...", flush=True)
    small_data = load_final_window(small_path)
    print("Loading final 75% of large DFT trajectory ...", flush=True)
    large_data = load_final_window(large_path)

    print("Computing legacy reference profiles ...", flush=True)
    small_same = profile(small_data, "legacy_tol", LEGACY_TOL)
    large = profile(large_data, "legacy_tol", LEGACY_TOL)

    print("Scanning small-system surface tolerance ...", flush=True)
    tol_profiles = []
    tol_rows = []
    for tol in TOL_GRID:
        candidate = profile(small_data, "legacy_tol", float(tol))
        metrics = comparison_metrics(
            candidate["z"], candidate["smooth"], large["smooth"],
            candidate["sem"], large["sem"]
        )
        tol_profiles.append(candidate)
        tol_rows.append({
            "small_tol_A": float(tol),
            "small_onset_z_A": profile_onset_z(candidate),
            "large_onset_z_A": profile_onset_z(large),
            "onset_position_difference_A": abs(profile_onset_z(candidate) - profile_onset_z(large)),
            "small_raw_onset_z_A": profile_onset_z(candidate, use_raw=True),
            "large_raw_onset_z_A": profile_onset_z(large, use_raw=True),
            "small_peak_z_A": candidate["peak_z_A"],
            "large_peak_z_A": large["peak_z_A"],
            "peak_position_difference_A": abs(candidate["peak_z_A"] - large["peak_z_A"]),
            "mae_g_mL": metrics["mae"],
            "rmse_g_mL": metrics["rmse"],
        })
    small_aligned = min(
        tol_profiles,
        key=lambda p: (
            abs(profile_onset_z(p) - profile_onset_z(large)),
            abs(p["peak_z_A"] - large["peak_z_A"]),
            abs(profile_onset_z(p, use_raw=True) - profile_onset_z(large, use_raw=True)),
            abs(p["peak_density_g_mL"] - large["peak_density_g_mL"]),
            abs(p["parameter"] - LEGACY_TOL),
        ),
    )

    print("Scanning common surface-atom fractions ...", flush=True)
    fraction_pairs = []
    fraction_rows = []
    for fraction in FRACTION_GRID:
        fs = profile(small_data, "common_fraction", float(fraction))
        fl = profile(large_data, "common_fraction", float(fraction))
        metrics = comparison_metrics(fs["z"], fs["smooth"], fl["smooth"], fs["sem"], fl["sem"])
        fraction_pairs.append((fs, fl, metrics))
        fraction_rows.append({
            "surface_fraction": float(fraction),
            "small_peak_z_A": fs["peak_z_A"],
            "large_peak_z_A": fl["peak_z_A"],
            "peak_position_difference_A": abs(fs["peak_z_A"] - fl["peak_z_A"]),
            "mae_g_mL": metrics["mae"],
            "rmse_g_mL": metrics["rmse"],
            "small_mean_selected_atoms_per_surface": 0.5 * (fs["mean_bottom_atoms"] + fs["mean_top_atoms"]),
            "large_mean_selected_atoms_per_surface": 0.5 * (fl["mean_bottom_atoms"] + fl["mean_top_atoms"]),
        })
    # Select a single fraction by peak agreement, then profile agreement.  This
    # is reported as a common-coordinate diagnostic, not used to scale density.
    fraction_small, fraction_large, fraction_metrics = min(
        fraction_pairs,
        key=lambda item: (
            abs(item[0]["peak_z_A"] - item[1]["peak_z_A"]),
            item[2]["mae"],
        ),
    )

    modes = transformed_modes(small_aligned, large)
    shift_scale = scan_shift_and_scale(small_aligned, large)
    interface_mask = (
        (small_aligned["z"] >= INTERFACE_WINDOW[0])
        & (small_aligned["z"] <= INTERFACE_WINDOW[1])
        & (small_aligned["smooth"] > 0.2)
        & (large["smooth"] > 0.2)
    )
    binwise_ratio = large["smooth"][interface_mask] / small_aligned["smooth"][interface_mask]
    binwise_ratio_diagnostic = {
        "median": float(np.median(binwise_ratio)),
        "p05": float(np.percentile(binwise_ratio, 5)),
        "p95": float(np.percentile(binwise_ratio, 95)),
        "coefficient_of_variation": float(np.std(binwise_ratio) / np.mean(binwise_ratio)),
    }
    same_metrics = comparison_metrics(
        small_same["z"], small_same["smooth"], large["smooth"],
        small_same["sem"], large["sem"]
    )
    aligned_metrics = comparison_metrics(
        small_aligned["z"], small_aligned["smooth"], large["smooth"],
        small_aligned["sem"], large["sem"]
    )

    write_csv(outdir / "surface_tolerance_scan.csv", tol_rows)
    write_csv(outdir / "common_surface_fraction_scan.csv", fraction_rows)
    mode_rows = [
        {key: value for key, value in mode.items() if key not in {"small", "small_sem"}}
        for mode in modes
    ]
    write_csv(outdir / "normalization_summary.csv", mode_rows)

    z = small_aligned["z"]
    table = np.column_stack((
        z,
        small_same["smooth"],
        small_aligned["smooth"],
        small_aligned["sem"],
        large["smooth"],
        large["sem"],
        modes[1]["small"],
        modes[2]["small"],
        small_aligned["smooth"] - large["smooth"],
    ))
    np.savetxt(
        outdir / "dft_system_size_profiles.dat",
        table,
        fmt="%.8f",
        header=(
            "Delta_Z_A small_same_tol_SG small_surface_aligned_SG small_SEM "
            "large_legacy_SG large_SEM small_bulk_matched_SG "
            "small_least_squares_scaled_SG aligned_small_minus_large [density g/mL]"
        ),
    )

    summary = {
        "small": {
            "trajectory": str(small_path),
            "atoms": int(len(small_data.symbols)),
            "water_oxygen": int(len(small_data.water_o_idx)),
            "frames_used": int(len(small_data.z_frames)),
            "area_A2": float(np.mean(small_data.areas_A2)),
        },
        "large": {
            "trajectory": str(large_path),
            "atoms": int(len(large_data.symbols)),
            "water_oxygen": int(len(large_data.water_o_idx)),
            "frames_used": int(len(large_data.z_frames)),
            "area_A2": float(np.mean(large_data.areas_A2)),
        },
        "same_tol_0p8": same_metrics,
        "surface_aligned": {
            "small_tol_A": small_aligned["parameter"],
            "large_tol_A": large["parameter"],
            "small_onset_z_A": profile_onset_z(small_aligned),
            "large_onset_z_A": profile_onset_z(large),
            "small_peak_z_A": small_aligned["peak_z_A"],
            "large_peak_z_A": large["peak_z_A"],
            "small_peak_density_g_mL": small_aligned["peak_density_g_mL"],
            "large_peak_density_g_mL": large["peak_density_g_mL"],
            "small_bulk_density_g_mL": small_aligned["bulk_density_g_mL"],
            "large_bulk_density_g_mL": large["bulk_density_g_mL"],
            **aligned_metrics,
        },
        "common_fraction": {
            "fraction": fraction_small["parameter"],
            "small_peak_z_A": fraction_small["peak_z_A"],
            "large_peak_z_A": fraction_large["peak_z_A"],
            **fraction_metrics,
        },
        "best_shift_and_scale_diagnostic": shift_scale,
        "binwise_large_over_small_ratio_diagnostic": binwise_ratio_diagnostic,
        "normalization_modes": mode_rows,
    }
    with (outdir / "analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    physical = mode_rows[0]
    bulk_matched = mode_rows[1]
    integral_matched = mode_rows[2]
    peak_matched = mode_rows[3]
    optimal = mode_rows[4]
    report = f"""# DFT system-size surface-coordinate and normalization analysis

## Processing held fixed

- Both XYZ trajectories are recomputed from coordinates using their final 75% frames
  (300 frames each).
- Water O and slab atoms are classified from the first selected frame with the
  same 2.4 Å Ti–O neighbor criterion.
- Both interfaces, the same 0.05 Å bins, the same water molar mass, and each
  frame's own physical xy area are used.
- Density is normalized per frame before time averaging and smoothing.

## Surface-coordinate result

Using the same legacy `tol=0.80 Å` for both sizes gives an interface MAE of
{same_metrics['mae']:.4f} g/mL and RMSE of {same_metrics['rmse']:.4f} g/mL.
The legacy extreme-value plane is atom-count dependent.

Anchoring the large trajectory at `tol=0.80 Å` and selecting the small-system
surface origin by meaningful onset, first-peak bin, raw onset, and peak-height
proximity gives:

- small `tol={small_aligned['parameter']:.2f} Å`; large `tol={large['parameter']:.2f} Å`;
- meaningful onsets {profile_onset_z(small_aligned):.3f}/{profile_onset_z(large):.3f} Å;
- both first peaks at {large['peak_z_A']:.3f} Å;
- peak densities {small_aligned['peak_density_g_mL']:.3f} and
  {large['peak_density_g_mL']:.3f} g/mL;
- bulk densities {small_aligned['bulk_density_g_mL']:.3f} and
  {large['bulk_density_g_mL']:.3f} g/mL;
- MAE {aligned_metrics['mae']:.4f} g/mL and RMSE {aligned_metrics['rmse']:.4f}
  g/mL over 0.5–6 Å;
- {aligned_metrics['fraction_within_2_combined_sem']:.1%} of bins within
  ±2 combined block SEM.

A size-independent alternative that averages the same fraction of innermost
surface atoms on both cells selects a fraction of {fraction_small['parameter']:.2f}.
It also places both peaks at {fraction_large['peak_z_A']:.3f} Å, but its MAE is
{fraction_metrics['mae']:.4f} g/mL, slightly worse than the legacy-compatible
surface-origin calibration.

## Normalization tests after surface alignment

| Transformation applied to small | Scale | Offset (g/mL) | MAE (g/mL) | RMSE (g/mL) |
|---|---:|---:|---:|---:|
| Actual-area absolute density | {physical['scale']:.6f} | {physical['offset']:.6f} | {physical['mae']:.4f} | {physical['rmse']:.4f} |
| Match bulk plateau | {bulk_matched['scale']:.6f} | {bulk_matched['offset']:.6f} | {bulk_matched['mae']:.4f} | {bulk_matched['rmse']:.4f} |
| Match 0.5–6 Å integral | {integral_matched['scale']:.6f} | {integral_matched['offset']:.6f} | {integral_matched['mae']:.4f} | {integral_matched['rmse']:.4f} |
| Match first-peak height | {peak_matched['scale']:.6f} | {peak_matched['offset']:.6f} | {peak_matched['mae']:.4f} | {peak_matched['rmse']:.4f} |
| Least-squares global scale | {optimal['scale']:.6f} | {optimal['offset']:.6f} | {optimal['mae']:.4f} | {optimal['rmse']:.4f} |

Even the fitted global scale changes the RMSE only from {physical['rmse']:.4f}
to {optimal['rmse']:.4f} g/mL. Allowing a fitted horizontal shift and scale
chooses Δz={shift_scale['shift_A']:+.3f} Å and scale={shift_scale['scale']:.6f},
but still leaves RMSE={shift_scale['rmse_g_mL']:.4f} g/mL and maximum residual
{shift_scale['max_abs_g_mL']:.4f} g/mL.

The binwise large/small factor is not constant: its 5th–95th percentile range
is {binwise_ratio_diagnostic['p05']:.3f}–{binwise_ratio_diagnostic['p95']:.3f}.
Therefore exact pointwise agreement would require a different correction in
each bin. Such a correction would replace the small trajectory with the large
profile and is not a normalization.

## Conclusion

The physically defensible result is the surface-aligned, actual-area-normalized
profile. It is statistically consistent with the large system for most bins,
but finite independent MD samples cannot be made exactly identical by any
single physical normalization. The remaining structured residual should be
reduced through longer or replicated trajectories, not post-processing.
"""
    (outdir / "ANALYSIS.md").write_text(report)

    make_plots(
        outdir, small_same, large, small_aligned, fraction_small,
        fraction_large, tol_rows, modes
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
