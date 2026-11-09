#!/usr/bin/env python3
"""Compare new and previous large-cell CACELES structure.

For MACE-MP-0(L), UMA-M(OC20), and PET-OMATPES, this script compares the
new trajectories under ``Large_CACELES`` with the previous large-system
trajectories under ``MDtraj_all_students``.  Both datasets use their final 75%
of saved frames, the same two-interface density normalization, and the local
inward-normal water orientation used by panel b of
``variants_diagnostic.py``.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "Water-dipole-analyse"
DENSITY_HOME = ROOT / "water_density_analysis"
PREVIOUS_ROOT_CANDIDATES = (
    Path("/Volumes/Extreme Pro/TiO2_water/MD_simulations/MDtraj_all_students"),
    Path("/Volumes/Extreme Pro 1/TiO2_water/MD_simulations/MDtraj_all_students"),
)
PREVIOUS_ROOT = next(
    (path for path in PREVIOUS_ROOT_CANDIDATES if path.is_dir()),
    PREVIOUS_ROOT_CANDIDATES[0],
)
NEW_ROOT = ROOT / "Large_CACELES"
RESULTS_DIR = HOME / "results" / "new_previous_large_caceles"
CACHE_DIR = HOME / "new_previous_large_caceles_cache"
OUTPUT_PNG = RESULTS_DIR / "new_previous_large_density_orientation.png"
OUTPUT_PDF = RESULTS_DIR / "new_previous_large_density_orientation.pdf"
OUTPUT_PROFILES = RESULTS_DIR / "new_previous_large_profiles.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "new_previous_large_summary.csv"

sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(DENSITY_HOME))
from trajectory_io import (  # noqa: E402
    open_binary,
    open_text,
    resolve_compressed,
    source_identity,
    uncompressed_size,
)
import analyze_dft_system_size_normalization as core  # noqa: E402


KEEP_FRACTION = 0.75
SURFACE_TOLERANCE = 0.80
DENSITY_BIN = 0.05
DENSITY_MAX = 12.0
ORIENTATION_BIN = 0.20
ORIENTATION_MAX = 15.0
NBLOCKS = 5
MIN_ORIENTATION_COUNT = 50
CACHE_VERSION = 4
INTERFACE_WINDOW = (0.6, 6.0)
BULK_WINDOW = (6.0, 10.0)


@dataclass(frozen=True)
class ModelPair:
    key: str
    label: str
    color: str
    previous_relative: str
    new_relative: str
    new_initial_structure: str

    @property
    def previous_path(self) -> Path:
        return resolve_compressed(PREVIOUS_ROOT / self.previous_relative)

    @property
    def new_path(self) -> Path:
        return resolve_compressed(NEW_ROOT / self.new_relative)


MODELS = [
    ModelPair(
        "mace_mp0",
        r"MACE-MP-0(L)",
        "#071d88",
        "MACE-MP0a_L/md_out-2025-12.xyz",
        "CACELES_MACE_MP0a_L/md_out.xyz",
        "PET-OMATPES previous last frame",
    ),
    ModelPair(
        "uma_oc20",
        r"UMA-M(OC20)",
        "#D81B60",
        "UMA-M-OC20/md_out-2025-12.xyz",
        "CACELES_UMA_OC20/md_out.xyz",
        "PET-OMATPES previous last frame",
    ),
    ModelPair(
        "pet_omatpes",
        r"PET-OMATPES (r$^2$SCAN)",
        "#E76F51",
        "PET-omatpes/md_out.xyz",
        "CACELES_PET_omat/md_out.xyz",
        "MACE-MP-0(L) previous last frame",
    ),
]

VERSIONS = ("previous", "new")

# The new jobs used a 1 fs timestep, 50,000-step warmup, and 100,000-step
# production.  The XYZ contains the 100 ps production only, saved every 500 fs.
NEW_WARMUP_PS = 50.0
NEW_PRODUCTION_PS = 100.0
NEW_XYZ_INTERVAL_PS = 0.5


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


def selected_frames(
    path: Path,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Yield positions, symbols, box lengths, and xy area for final 75%."""
    natoms, nframes = trajectory_shape(path)
    keep = int(np.floor(nframes * KEEP_FRACTION))
    start = nframes - keep
    symbols: np.ndarray | None = None

    path = resolve_compressed(path)
    with open_text(path) as handle:
        for frame_index in range(nframes):
            this_natoms = int(handle.readline())
            comment = handle.readline()
            if this_natoms != natoms:
                raise ValueError(f"{path}: atom count changed at frame {frame_index}")
            if frame_index < start:
                for _ in range(natoms):
                    handle.readline()
                continue

            cell = core.parse_lattice(comment)
            if not np.allclose(cell, np.diag(np.diag(cell)), atol=1.0e-8):
                raise ValueError(f"{path}: expected an orthorhombic cell")
            box = np.diag(cell).copy()
            area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
            positions = np.empty((natoms, 3), dtype=np.float64)
            if symbols is None:
                symbols = np.empty(natoms, dtype="U2")
            for atom_index in range(natoms):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(
                        f"{path}: malformed atom row in frame {frame_index}"
                    )
                if frame_index == start:
                    symbols[atom_index] = fields[0]
                positions[atom_index] = [float(v) for v in fields[1:4]]
            yield positions, symbols, box, area


class ProfileAccumulator:
    def __init__(self) -> None:
        self.density_edges = np.arange(
            0.0, DENSITY_MAX + 0.5 * DENSITY_BIN, DENSITY_BIN
        )
        self.orientation_edges = np.arange(
            0.0, ORIENTATION_MAX + 0.5 * ORIENTATION_BIN, ORIENTATION_BIN
        )
        self.density_counts: list[np.ndarray] = []
        self.orientation_sums: list[np.ndarray] = []
        self.orientation_counts: list[np.ndarray] = []
        self.areas: list[float] = []
        self.half_gaps: list[float] = []
        self.dissociation: list[float] = []
        self.slab_idx: np.ndarray | None = None
        self.water_o_idx: np.ndarray | None = None
        self.o_idx: np.ndarray | None = None
        self.h_idx: np.ndarray | None = None
        self.water_o_mask: np.ndarray | None = None

    def initialize(
        self, positions: np.ndarray, symbols: np.ndarray, box: np.ndarray
    ) -> None:
        slab_idx, water_o_idx = core.classify_atoms(
            symbols, positions, np.diag(box)
        )
        o_idx = np.flatnonzero(symbols == "O")
        self.slab_idx = slab_idx
        self.water_o_idx = water_o_idx
        self.o_idx = o_idx
        self.h_idx = np.flatnonzero(symbols == "H")
        is_water = np.zeros(len(symbols), dtype=bool)
        is_water[water_o_idx] = True
        self.water_o_mask = is_water[o_idx]

    def add(
        self,
        positions: np.ndarray,
        symbols: np.ndarray,
        box: np.ndarray,
        area: float,
    ) -> None:
        if self.slab_idx is None:
            self.initialize(positions, symbols, box)
        assert self.slab_idx is not None
        assert self.water_o_idx is not None
        assert self.o_idx is not None
        assert self.h_idx is not None
        assert self.water_o_mask is not None

        bottom, top, _, _ = core.surface_planes(
            positions[self.slab_idx, 2], "legacy_tol", SURFACE_TOLERANCE
        )
        midpoint = 0.5 * (bottom + top)
        self.half_gaps.append(0.5 * (top - bottom))

        # Standard density definition: all O atoms classified as water oxygen.
        water_z = positions[self.water_o_idx, 2]
        density_distance = np.where(
            water_z <= midpoint, water_z - bottom, top - water_z
        )
        self.density_counts.append(
            np.histogram(density_distance, bins=self.density_edges)[0]
        )
        self.areas.append(area)

        # Orientation definition used by variants_diagnostic.py: a water O is
        # intact when exactly two H atoms have it as their nearest periodic O.
        tree = cKDTree(np.mod(positions[self.o_idx], box), boxsize=box)
        _, owner = tree.query(np.mod(positions[self.h_idx], box))
        owner_counts = np.bincount(owner, minlength=len(self.o_idx))
        intact = np.flatnonzero(self.water_o_mask & (owner_counts == 2))
        self.dissociation.append(
            1.0 - intact.size / max(int(np.sum(self.water_o_mask)), 1)
        )

        order = np.argsort(owner, kind="stable")
        sorted_owner = owner[order]
        first = np.searchsorted(sorted_owner, intact, side="left")
        oxygen = positions[self.o_idx[intact]]
        h1 = self.h_idx[order[first]]
        h2 = self.h_idx[order[first + 1]]
        d1 = positions[h1] - oxygen
        d2 = positions[h2] - oxygen
        d1 -= box * np.round(d1 / box)
        d2 -= box * np.round(d2 / box)
        bisector = 0.5 * (d1 + d2)
        norm = np.linalg.norm(bisector, axis=1)
        lower = oxygen[:, 2] <= midpoint
        orientation_distance = np.where(
            lower, oxygen[:, 2] - bottom, top - oxygen[:, 2]
        )
        # Panel-b convention: project onto the local inward surface normal.
        # The upper-interface sign is reversed before combining both sides.
        inward_component = np.where(lower, bisector[:, 2], -bisector[:, 2])
        cos_theta = inward_component / np.maximum(norm, 1.0e-12)
        self.orientation_sums.append(
            np.histogram(
                orientation_distance,
                bins=self.orientation_edges,
                weights=cos_theta,
            )[0]
        )
        self.orientation_counts.append(
            np.histogram(orientation_distance, bins=self.orientation_edges)[0]
        )

    def finalize(self) -> dict[str, np.ndarray | float | int]:
        density_counts = np.asarray(self.density_counts, dtype=float)
        orientation_sums = np.asarray(self.orientation_sums, dtype=float)
        orientation_counts = np.asarray(self.orientation_counts, dtype=float)
        areas = np.asarray(self.areas, dtype=float)
        per_frame_density = (
            density_counts
            / (2.0 * areas[:, None] * DENSITY_BIN)
            * 18.01528
            / 6.02214076e23
            * 1.0e24
        )
        density_raw = per_frame_density.mean(axis=0)
        density = savgol_filter(density_raw, window_length=7, polyorder=3)

        total_orientation_sum = orientation_sums.sum(axis=0)
        total_orientation_count = orientation_counts.sum(axis=0)
        orientation = np.divide(
            total_orientation_sum,
            total_orientation_count,
            out=np.full(len(total_orientation_sum), np.nan),
            where=total_orientation_count >= MIN_ORIENTATION_COUNT,
        )

        density_blocks = []
        orientation_blocks = []
        for indices in np.array_split(np.arange(len(density_counts)), NBLOCKS):
            density_blocks.append(per_frame_density[indices].mean(axis=0))
            block_sum = orientation_sums[indices].sum(axis=0)
            block_count = orientation_counts[indices].sum(axis=0)
            orientation_blocks.append(
                np.divide(
                    block_sum,
                    block_count,
                    out=np.full(len(block_sum), np.nan),
                    where=block_count > 0,
                )
            )
        density_sem = np.std(np.asarray(density_blocks), axis=0, ddof=1) / np.sqrt(
            NBLOCKS
        )
        orientation_sem = finite_sem(np.asarray(orientation_blocks))
        orientation_sem[total_orientation_count < MIN_ORIENTATION_COUNT] = np.nan
        return {
            "density_z": 0.5 * (self.density_edges[:-1] + self.density_edges[1:]),
            "density_raw": density_raw,
            "density": density,
            "density_sem": density_sem,
            "orientation_z": 0.5
            * (self.orientation_edges[:-1] + self.orientation_edges[1:]),
            "orientation": orientation,
            "orientation_sem": orientation_sem,
            "orientation_count": total_orientation_count,
            "frames": len(density_counts),
            "mean_area_A2": float(np.mean(areas)),
            "mean_half_gap_A": float(np.mean(self.half_gaps)),
            "mean_dissociation_fraction": float(np.mean(self.dissociation)),
        }


def finite_sem(values: np.ndarray) -> np.ndarray:
    valid = np.sum(np.isfinite(values), axis=0)
    mean = np.divide(
        np.nansum(values, axis=0),
        valid,
        out=np.full(values.shape[1], np.nan),
        where=valid > 0,
    )
    squared = np.nansum((values - mean[None, :]) ** 2, axis=0)
    variance = np.divide(
        squared,
        valid - 1,
        out=np.full(values.shape[1], np.nan),
        where=valid > 1,
    )
    return np.sqrt(variance / valid)


def cache_path(model: ModelPair, version: str) -> Path:
    return CACHE_DIR / f"{model.key}_{version}.npz"


def source_path(model: ModelPair, version: str) -> Path:
    if version == "previous":
        return model.previous_path
    if version == "new":
        return model.new_path
    raise ValueError(f"{model.label}: dataset {version!r} is unavailable")


def cache_current(path: Path, source: Path) -> bool:
    if not path.is_file() or not source.is_file():
        return False
    stat = source.stat()
    with np.load(path, allow_pickle=False) as data:
        return bool(
            data["cache_version"].item() == CACHE_VERSION
            and data["source_size"].item() == uncompressed_size(source)
            and data["source_mtime_ns"].item() == stat.st_mtime_ns
            and data["keep_fraction"].item() == KEEP_FRACTION
        )


def save_cache(
    path: Path,
    source: Path,
    result: dict[str, np.ndarray | float | int],
) -> None:
    source_path_value, source_size, source_mtime_ns = source_identity(source)
    np.savez_compressed(
        path,
        **result,
        cache_version=np.asarray(CACHE_VERSION),
        source_path=np.asarray(source_path_value),
        source_size=np.asarray(source_size),
        source_mtime_ns=np.asarray(source_mtime_ns),
        keep_fraction=np.asarray(KEEP_FRACTION),
    )


def load_cache(path: Path) -> dict[str, np.ndarray | float | int]:
    metadata = {
        "cache_version",
        "source_path",
        "source_size",
        "source_mtime_ns",
        "keep_fraction",
    }
    with np.load(path, allow_pickle=False) as data:
        return {
            key: data[key].copy() if data[key].ndim else data[key].item()
            for key in data.files
            if key not in metadata
        }


def analyze(model: ModelPair, version: str) -> dict[str, np.ndarray | float | int]:
    source = source_path(model, version)
    cache = cache_path(model, version)
    if cache_current(cache, source):
        print(f"{model.label} {version}: loading cache", flush=True)
        return load_cache(cache)
    if not source.is_file():
        raise FileNotFoundError(source)
    natoms, nframes = trajectory_shape(source)
    keep = int(np.floor(nframes * KEEP_FRACTION))
    print(
        f"{model.label} {version}: {natoms} atoms, {nframes} frames; "
        f"analyzing final {keep}",
        flush=True,
    )
    accumulator = ProfileAccumulator()
    for index, (positions, symbols, box, area) in enumerate(
        selected_frames(source), start=1
    ):
        accumulator.add(positions, symbols, box, area)
        if index % 100 == 0 or index == keep:
            print(f"  {index}/{keep} frames", flush=True)
    result = accumulator.finalize()
    save_cache(cache, source, result)
    return result


def interpolate_common(
    z_first: np.ndarray,
    y_first: np.ndarray,
    z_second: np.ndarray,
    y_second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = max(float(z_first.min()), float(z_second.min()))
    hi = min(float(z_first.max()), float(z_second.max()))
    keep = (z_first >= lo) & (z_first <= hi)
    z = z_first[keep]
    return z, y_first[keep], np.interp(z, z_second, y_second)


def metrics(
    model: ModelPair,
    previous: dict[str, np.ndarray | float | int],
    new: dict[str, np.ndarray | float | int],
) -> dict[str, object]:
    zd, old_d, new_d = interpolate_common(
        previous["density_z"],
        previous["density"],
        new["density_z"],
        new["density"],
    )
    zo, old_o, new_o = interpolate_common(
        previous["orientation_z"],
        previous["orientation"],
        new["orientation_z"],
        new["orientation"],
    )
    density_window = (zd >= INTERFACE_WINDOW[0]) & (zd <= INTERFACE_WINDOW[1])
    orientation_window = (zo >= INTERFACE_WINDOW[0]) & (zo <= 10.0)
    bulk = (zo >= BULK_WINDOW[0]) & (zo <= BULK_WINDOW[1])
    peak_window = (zd >= 0.8) & (zd <= 2.0)
    old_peak_index = np.flatnonzero(peak_window)[np.argmax(old_d[peak_window])]
    new_peak_index = np.flatnonzero(peak_window)[np.argmax(new_d[peak_window])]
    return {
        "model": model.key,
        "label": model.label,
        "previous_frames": int(previous["frames"]),
        "new_frames": int(new["frames"]),
        "new_warmup_ps": NEW_WARMUP_PS,
        "new_production_ps": NEW_PRODUCTION_PS,
        "new_analyzed_snapshot_span_ps": (
            max(int(new["frames"]) - 1, 0) * NEW_XYZ_INTERVAL_PS
        ),
        "new_initial_structure": model.new_initial_structure,
        "density_mae_0p6_6A_g_mL": float(
            np.mean(np.abs(new_d[density_window] - old_d[density_window]))
        ),
        "density_rmse_0p6_6A_g_mL": float(
            np.sqrt(np.mean((new_d[density_window] - old_d[density_window]) ** 2))
        ),
        "previous_first_peak_z_A": float(zd[old_peak_index]),
        "previous_first_peak_g_mL": float(old_d[old_peak_index]),
        "new_first_peak_z_A": float(zd[new_peak_index]),
        "new_first_peak_g_mL": float(new_d[new_peak_index]),
        "orientation_rmse_0p6_10A": float(
            np.sqrt(np.nanmean((new_o[orientation_window] - old_o[orientation_window]) ** 2))
        ),
        "previous_mean_cos_6_10A": float(np.nanmean(old_o[bulk])),
        "new_mean_cos_6_10A": float(np.nanmean(new_o[bulk])),
        "previous_dissociation_fraction": float(
            previous["mean_dissociation_fraction"]
        ),
        "new_dissociation_fraction": float(new["mean_dissociation_fraction"]),
    }


def write_outputs(
    results: dict[tuple[str, str], dict[str, np.ndarray | float | int]],
    summaries: list[dict[str, object]],
) -> None:
    with OUTPUT_PROFILES.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "label",
                "simulation",
                "observable",
                "distance_A",
                "value",
                "block_sem",
            ]
        )
        for model in MODELS:
            for version in VERSIONS:
                result = results[(model.key, version)]
                for z, value, sem in zip(
                    result["density_z"],
                    result["density"],
                    result["density_sem"],
                ):
                    writer.writerow(
                        [model.key, model.label, version, "density_g_mL", z, value, sem]
                    )
                for z, value, sem in zip(
                    result["orientation_z"],
                    result["orientation"],
                    result["orientation_sem"],
                ):
                    writer.writerow(
                        [model.key, model.label, version, "mean_cos_theta", z, value, sem]
                    )
    with OUTPUT_SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def make_figure(
    results: dict[tuple[str, str], dict[str, np.ndarray | float | int]]
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8.27, 5.2),
        dpi=500,
        sharex="col",
        sharey="row",
    )

    for column, model in enumerate(MODELS):
        density_ax = axes[0, column]
        orientation_ax = axes[1, column]
        for version, linestyle, color, linewidth in (
            ("previous", "-", model.color, 2.0),
            ("new", "--", "0.10", 2.0),
        ):
            result = results[(model.key, version)]
            density_ax.plot(
                result["density_z"],
                result["density"],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            orientation_ax.plot(
                result["orientation_z"],
                result["orientation"],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        density_ax.set_xlim(0.6, 10.0)
        density_ax.set_ylim(0.0, 5.0)
        orientation_ax.set_xlim(0.6, 10.0)
        orientation_ax.set_ylim(-1.0, 1.0)
        orientation_ax.axhline(0.0, color="0.55", linewidth=0.7)
        density_ax.text(
            0.5,
            0.96,
            model.label,
            transform=density_ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        for ax in (density_ax, orientation_ax):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.tick_params(top=False, right=False)
        density_ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        density_ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        orientation_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        orientation_ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes[0, 0].set_ylabel("Density [g/mL]", labelpad=3)
    axes[1, 0].set_ylabel(r"$\langle\cos\theta\rangle$", labelpad=3)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    legend_handles = [
        Line2D([], [], color="0.25", linestyle="-", linewidth=2.0),
        Line2D([], [], color="0.10", linestyle="--", linewidth=2.0),
    ]
    fig.legend(
        legend_handles,
        ["Previous large simulation", "New large simulation"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.supxlabel(
        r"Distance from $\mathrm{TiO}_2$ surface [$\mathrm{\AA}$]",
        x=0.54,
        y=0.015,
        fontsize=11,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.14,
        top=0.87,
        wspace=0.10,
        hspace=0.10,
    )
    fig.savefig(OUTPUT_PNG, dpi=500)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        (model.key, version): analyze(model, version)
        for model in MODELS
        for version in VERSIONS
    }
    summaries = [
        metrics(
            model,
            results[(model.key, "previous")],
            results[(model.key, "new")],
        )
        for model in MODELS
    ]
    write_outputs(results, summaries)
    make_figure(results)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PROFILES}")
    print(f"Saved {OUTPUT_SUMMARY}")
    for row in summaries:
        print(
            f"{row['label']}: density MAE={row['density_mae_0p6_6A_g_mL']:.3f} "
            f"g/mL; orientation RMSE={row['orientation_rmse_0p6_10A']:.3f}; "
            f"local-normal cos 6-10 A old/new="
            f"{row['previous_mean_cos_6_10A']:+.3f}/"
            f"{row['new_mean_cos_6_10A']:+.3f}"
        )


if __name__ == "__main__":
    main()
