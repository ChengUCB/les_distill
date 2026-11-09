#!/usr/bin/env python3
"""Plot water-dipole orientation for all large-cell CACELES students.

The three stacked panels and model classification follow
``finalplot_combined_density_ir.py`` (PBE, RPBE(-D3), and meta-GGA).  The
orientation definition follows panel (b) of ``variants_diagnostic.png``:
intact waters are reconstructed every frame, the molecular bisector points
from O toward the H midpoint, and cos(theta) is measured against the inward
normal from the nearest anatase TiO2(101) surface.  Thus cos(theta) > 0 means
that the H atoms point away from the surface and into the water film.
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
import zstandard
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1] / "small-Direct_MD_foundation"
HOME = ROOT / "Water-dipole-analyse"
DENSITY_HOME = ROOT / "water_density_analysis"
TRAJECTORY_ROOT = Path(
    "/Volumes/Extreme Pro/TiO2_water/MD_simulations/MDtraj_all_students"
)
RESULTS_DIR = HOME / "results"
SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "Cos-theta":
    RESULTS_DIR = SCRIPT_DIR
CACHE_DIR = HOME / "all_caceles_orientation_cache"
OUTPUT_PNG = RESULTS_DIR / "all_caceles_student_water_orientation.png"
OUTPUT_PDF = RESULTS_DIR / "Fig5_student_water_orientation.pdf"
OUTPUT_CSV = RESULTS_DIR / "all_caceles_student_water_orientation.csv"
OUTPUT_SUMMARY_CSV = (
    RESULTS_DIR / "all_caceles_student_water_orientation_summary.csv"
)

sys.path.insert(0, str(HOME))
sys.path.insert(0, str(DENSITY_HOME))

import analyze_dft_system_size_normalization as core  # noqa: E402
from analyze_water_dipole import load_full  # noqa: E402


KEEP_FRACTION = 0.75
SURFACE_TOLERANCE = 0.80
DZ_BIN = 0.20
DZ_MAX = 15.0
NBLOCKS = 5
MIN_TOTAL_BIN_COUNT = 50
CACHE_VERSION = 1
FINETUNED_CASE_KEYS = {"mace_mp_ft10", "mace_mp_ft50"}
FINETUNED_COLOR = "#D62728"


@dataclass(frozen=True)
class Case:
    key: str
    relative_path: str
    label: str
    group: str
    color: str
    linestyle: object
    legacy_cache_name: str | None = None

    @property
    def path(self) -> Path:
        return TRAJECTORY_ROOT / self.relative_path


CASES = [
    # PBE
    Case(
        "mace_mp0",
        "MACE-MP0a_L/md_out-2025-12.xyz",
        r"MACE-MP-0(L)",
        "PBE",
        "#071d88",
        "-",
        "MP_large",
    ),
    Case(
        "mace_mh_omat",
        "MACE-MH-omat/md_out-2025-12.xyz",
        r"MACE-MH-1(Omat)",
        "PBE",
        "#87CEEB",
        "--",
        "MH_large",
    ),
    Case(
        "gemnet_oc22",
        "GemNet-OC22/md_out.xyz",
        r"GemNet-OC22",
        "PBE",
        "#8f53fe",
        "-.",
        "GemNet_large",
    ),
    # RPBE(-D3)
    Case(
        "uma_m_oc20",
        "UMA-M-OC20/md_out-2025-12.xyz",
        r"UMA-M(OC20)",
        "RPBE",
        "#D81B60",
        "-",
    ),
    Case(
        "esen_sm_con",
        "esen-sm-con/md_out-2025-12.xyz",
        r"eSEN-OC25-sm$^{\mathbf{\dagger}}$",
        "RPBE",
        "#FF8C00",
        "--",
    ),
    Case(
        "esen_direct",
        "esen-direct/md_out-2025-12.xyz.zst",
        r"eSEN-OC25-md$^{\mathbf{\dagger}}$",
        "RPBE",
        "#fd8bf5",
        "-.",
    ),
    # SCAN/meta-GGA
    Case(
        "caceles_scan_dft",
        "CACELES-SCAN-DFT/md_out.xyz",
        r"CACELES-SCAN",
        "SCAN",
        "#ffb81d",
        "-",
        "SCAN_student_large",
    ),
    Case(
        "mace_mp_ft10",
        "MACE-MP-finetuning-10/md_out_finetuning_10.xyz",
        r"MACE-MP-0(L) $\rightarrow$ 10% SCAN dataset",
        "SCAN",
        "#1F9E89",
        "-.",
        "MP_finetuned_10_large",
    ),
    Case(
        "mace_mp_ft50",
        "MACE-MP-finetuning-50/md_out_finetuning_50.xyz",
        r"MACE-MP-0(L) $\rightarrow$ 50% SCAN dataset",
        "SCAN",
        "#73D055",
        ":",
        "MP_finetuned_50_large",
    ),
    Case(
        "mace_h1_r2scan",
        "MACE-H1-r2scan/md_out.xyz",
        r"MACE-MH-1(r$^2$SCAN)",
        "SCAN",
        "#1d65a6",
        (0, (3, 1, 1, 1)),
    ),
    Case(
        "pet_omatpes",
        "PET-omatpes/md_out.xyz",
        r"PET-OMATPES (r$^2$SCAN)",
        "SCAN",
        "#E76F51",
        (0, (1, 1)),
    ),
]

GROUPS = [
    ("PBE", "PBE"),
    ("RPBE", "RPBE(-D3)"),
    ("SCAN", "meta-GGA"),
]
GROUP_LABELS = dict(GROUPS)


def open_text(path: Path):
    """Open plain or zstd-compressed extended XYZ as text."""
    if path.suffix == ".zst":
        return zstandard.open(path, mode="rt")
    return path.open()


def open_binary(path: Path):
    """Open plain or zstd-compressed extended XYZ as bytes."""
    if path.suffix == ".zst":
        return zstandard.open(path, mode="rb")
    return path.open("rb")


def line_count(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    count = 0
    with open_binary(path) as handle:
        while chunk := handle.read(chunk_size):
            count += chunk.count(b"\n")
    return count


def trajectory_shape(path: Path) -> tuple[int, int]:
    with open_text(path) as handle:
        natoms = int(handle.readline())
    frame_lines = natoms + 2
    nlines = line_count(path)
    if nlines % frame_lines:
        raise ValueError(f"{path}: incomplete XYZ trajectory")
    return natoms, nlines // frame_lines


def selected_frame_iterator(
    path: Path,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield final-75% positions, symbols, and orthorhombic box lengths."""
    natoms, nframes = trajectory_shape(path)
    keep = int(np.floor(nframes * KEEP_FRACTION))
    start = nframes - keep
    symbols: np.ndarray | None = None

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
                raise ValueError(f"{path}: orientation code requires an orthorhombic cell")
            box = np.diag(cell).copy()
            positions = np.empty((natoms, 3), dtype=np.float64)
            if symbols is None:
                symbols = np.empty(natoms, dtype="U2")
            for atom_index in range(natoms):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(
                        f"{path}: malformed atom line in frame {frame_index}"
                    )
                if frame_index == start:
                    symbols[atom_index] = fields[0]
                positions[atom_index] = [float(v) for v in fields[1:4]]
            yield positions, symbols, box


class OrientationAccumulator:
    """Accumulate folded intact-water orientation frame by frame."""

    def __init__(self) -> None:
        self.edges = np.arange(0.0, DZ_MAX + 0.5 * DZ_BIN, DZ_BIN)
        self.z = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.frame_sums: list[np.ndarray] = []
        self.frame_counts: list[np.ndarray] = []
        self.slab_idx: np.ndarray | None = None
        self.o_idx: np.ndarray | None = None
        self.h_idx: np.ndarray | None = None
        self.water_o_mask: np.ndarray | None = None
        self.dissociation: list[float] = []

    def initialize(
        self, positions: np.ndarray, symbols: np.ndarray, box: np.ndarray
    ) -> None:
        slab_idx, water_o_idx = core.classify_atoms(
            symbols, positions, np.diag(box)
        )
        o_idx = np.flatnonzero(symbols == "O")
        self.slab_idx = slab_idx
        self.o_idx = o_idx
        self.h_idx = np.flatnonzero(symbols == "H")
        is_water = np.zeros(len(symbols), dtype=bool)
        is_water[water_o_idx] = True
        self.water_o_mask = is_water[o_idx]

    def add(
        self, positions: np.ndarray, symbols: np.ndarray, box: np.ndarray
    ) -> None:
        if self.slab_idx is None:
            self.initialize(positions, symbols, box)
        assert self.slab_idx is not None
        assert self.o_idx is not None
        assert self.h_idx is not None
        assert self.water_o_mask is not None

        slab_z = positions[self.slab_idx, 2]
        bottom, top, _, _ = core.surface_planes(
            slab_z, "legacy_tol", SURFACE_TOLERANCE
        )
        midpoint = 0.5 * (bottom + top)

        tree = cKDTree(np.mod(positions[self.o_idx], box), boxsize=box)
        _, nearest_owner = tree.query(np.mod(positions[self.h_idx], box))
        owner_counts = np.bincount(nearest_owner, minlength=len(self.o_idx))
        intact = np.flatnonzero(self.water_o_mask & (owner_counts == 2))
        self.dissociation.append(
            1.0 - intact.size / max(int(np.sum(self.water_o_mask)), 1)
        )

        order = np.argsort(nearest_owner, kind="stable")
        sorted_owner = nearest_owner[order]
        first = np.searchsorted(sorted_owner, intact, side="left")
        oxygen_positions = positions[self.o_idx[intact]]
        h1 = self.h_idx[order[first]]
        h2 = self.h_idx[order[first + 1]]
        d1 = positions[h1] - oxygen_positions
        d2 = positions[h2] - oxygen_positions
        d1 -= box * np.round(d1 / box)
        d2 -= box * np.round(d2 / box)

        bisector = 0.5 * (d1 + d2)
        norm = np.linalg.norm(bisector, axis=1)
        lower_side = oxygen_positions[:, 2] <= midpoint
        distance = np.where(
            lower_side,
            oxygen_positions[:, 2] - bottom,
            top - oxygen_positions[:, 2],
        )
        inward_component = np.where(
            lower_side, bisector[:, 2], -bisector[:, 2]
        )
        cos_theta = inward_component / np.maximum(norm, 1.0e-12)

        frame_sum = np.histogram(
            distance, bins=self.edges, weights=cos_theta
        )[0].astype(float)
        frame_count = np.histogram(distance, bins=self.edges)[0].astype(float)
        self.frame_sums.append(frame_sum)
        self.frame_counts.append(frame_count)

    def finalize(self) -> dict[str, np.ndarray | float | int]:
        sums = np.asarray(self.frame_sums)
        counts = np.asarray(self.frame_counts)
        total_sums = sums.sum(axis=0)
        total_counts = counts.sum(axis=0)
        mean = np.divide(
            total_sums,
            total_counts,
            out=np.full_like(total_sums, np.nan),
            where=total_counts >= MIN_TOTAL_BIN_COUNT,
        )

        block_profiles = []
        for indices in np.array_split(np.arange(len(sums)), NBLOCKS):
            block_sum = sums[indices].sum(axis=0)
            block_count = counts[indices].sum(axis=0)
            block_profiles.append(
                np.divide(
                    block_sum,
                    block_count,
                    out=np.full_like(block_sum, np.nan),
                    where=block_count > 0,
                )
            )
        block_array = np.asarray(block_profiles)
        valid_blocks = np.sum(np.isfinite(block_array), axis=0)
        block_mean = np.divide(
            np.nansum(block_array, axis=0),
            valid_blocks,
            out=np.full(block_array.shape[1], np.nan),
            where=valid_blocks > 0,
        )
        squared_deviation = np.nansum(
            (block_array - block_mean[None, :]) ** 2, axis=0
        )
        sample_variance = np.divide(
            squared_deviation,
            valid_blocks - 1,
            out=np.full(block_array.shape[1], np.nan),
            where=valid_blocks > 1,
        )
        sem = np.sqrt(sample_variance / valid_blocks)
        sem[total_counts < MIN_TOTAL_BIN_COUNT] = np.nan
        return {
            "z": self.z,
            "orientation": mean,
            "sem": sem,
            "count": total_counts,
            "frames": len(sums),
            "dissociation_fraction": float(np.mean(self.dissociation)),
        }


def analyze_position_array(
    positions: np.ndarray, symbols: np.ndarray, box: np.ndarray
) -> dict[str, np.ndarray | float | int]:
    accumulator = OrientationAccumulator()
    for frame in positions:
        accumulator.add(frame.astype(float), symbols, np.asarray(box, dtype=float))
    return accumulator.finalize()


def analyze_stream(path: Path) -> dict[str, np.ndarray | float | int]:
    accumulator = OrientationAccumulator()
    for positions, symbols, box in selected_frame_iterator(path):
        accumulator.add(positions, symbols, box)
    return accumulator.finalize()


def cache_is_current(cache_path: Path, source_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    stat = source_path.stat()
    with np.load(cache_path, allow_pickle=False) as data:
        return bool(
            data.get("cache_version", -1) == CACHE_VERSION
            and data.get("source_size", -1) == stat.st_size
            and data.get("source_mtime_ns", -1) == stat.st_mtime_ns
            and data.get("keep_fraction", -1.0) == KEEP_FRACTION
            and data.get("dz_bin", -1.0) == DZ_BIN
            and data.get("surface_tolerance", -1.0) == SURFACE_TOLERANCE
        )


def load_result_cache(cache_path: Path) -> dict[str, np.ndarray | float | int]:
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "z": data["z"],
            "orientation": data["orientation"],
            "sem": data["sem"],
            "count": data["count"],
            "frames": int(data["frames"]),
            "dissociation_fraction": float(data["dissociation_fraction"]),
        }


def save_result_cache(
    cache_path: Path,
    source_path: Path,
    result: dict[str, np.ndarray | float | int],
) -> None:
    stat = source_path.stat()
    np.savez_compressed(
        cache_path,
        **result,
        cache_version=np.asarray(CACHE_VERSION),
        source_size=np.asarray(stat.st_size),
        source_mtime_ns=np.asarray(stat.st_mtime_ns),
        keep_fraction=np.asarray(KEEP_FRACTION),
        dz_bin=np.asarray(DZ_BIN),
        surface_tolerance=np.asarray(SURFACE_TOLERANCE),
    )


def calculate_case(case: Case) -> dict[str, np.ndarray | float | int]:
    cache_path = CACHE_DIR / f"{case.key}.npz"
    if not case.path.is_file():
        if cache_path.is_file():
            print(
                f"Loading saved orientation cache (source unavailable): "
                f"{case.label}",
                flush=True,
            )
            return load_result_cache(cache_path)
        raise FileNotFoundError(case.path)
    if cache_is_current(cache_path, case.path):
        print(f"Loading cached orientation: {case.label}", flush=True)
        return load_result_cache(cache_path)

    print(f"Analyzing orientation: {case.label}", flush=True)
    if case.legacy_cache_name is not None:
        positions, symbols, box = load_full(case.legacy_cache_name)
        result = analyze_position_array(positions, symbols, box)
    else:
        result = analyze_stream(case.path)
    save_result_cache(cache_path, case.path, result)
    return result


def write_csv(results: dict[str, dict[str, np.ndarray | float | int]]) -> None:
    """Write a tidy, machine-readable profile table and one-row/model summary."""
    with OUTPUT_CSV.open("w", newline="") as handle:
        fieldnames = [
            "group_order",
            "model_order",
            "group_code",
            "group_label",
            "model_key",
            "model_label",
            "source_trajectory",
            "distance_A",
            "mean_cos_theta",
            "block_sem",
            "molecule_samples",
            "profile_status",
            "frames_analyzed",
            "keep_fraction",
            "bin_width_A",
            "mean_dissociation_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_order, case in enumerate(CASES, start=1):
            result = results[case.key]
            for z, mean, sem, count in zip(
                result["z"],
                result["orientation"],
                result["sem"],
                result["count"],
            ):
                writer.writerow(
                    {
                        "group_order": next(
                            index
                            for index, (code, _) in enumerate(GROUPS, start=1)
                            if code == case.group
                        ),
                        "model_order": model_order,
                        "group_code": case.group,
                        "group_label": GROUP_LABELS[case.group],
                        "model_key": case.key,
                        "model_label": case.label,
                        "source_trajectory": str(case.path),
                        "distance_A": f"{float(z):.6f}",
                        "mean_cos_theta": (
                            f"{float(mean):.8f}" if np.isfinite(mean) else ""
                        ),
                        "block_sem": (
                            f"{float(sem):.8f}" if np.isfinite(sem) else ""
                        ),
                        "molecule_samples": int(count),
                        "profile_status": (
                            "valid"
                            if int(count) >= MIN_TOTAL_BIN_COUNT
                            else "insufficient_samples"
                        ),
                        "frames_analyzed": int(result["frames"]),
                        "keep_fraction": f"{KEEP_FRACTION:.2f}",
                        "bin_width_A": f"{DZ_BIN:.2f}",
                        "mean_dissociation_fraction": (
                            f"{float(result['dissociation_fraction']):.8f}"
                        ),
                    }
                )

    with OUTPUT_SUMMARY_CSV.open("w", newline="") as handle:
        fieldnames = [
            "group_order",
            "model_order",
            "group_code",
            "group_label",
            "model_key",
            "model_label",
            "source_trajectory",
            "frames_analyzed",
            "keep_fraction",
            "bin_width_A",
            "mean_dissociation_fraction",
            "mean_cos_theta_6_10A",
            "molecule_samples_6_10A",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_order, case in enumerate(CASES, start=1):
            result = results[case.key]
            z = np.asarray(result["z"], dtype=float)
            orientation = np.asarray(result["orientation"], dtype=float)
            count = np.asarray(result["count"], dtype=float)
            bulk = (
                (z >= 6.0)
                & (z < 10.0)
                & np.isfinite(orientation)
                & (count > 0)
            )
            bulk_samples = int(np.sum(count[bulk]))
            bulk_mean = (
                float(np.average(orientation[bulk], weights=count[bulk]))
                if bulk_samples
                else float("nan")
            )
            writer.writerow(
                {
                    "group_order": next(
                        index
                        for index, (code, _) in enumerate(GROUPS, start=1)
                        if code == case.group
                    ),
                    "model_order": model_order,
                    "group_code": case.group,
                    "group_label": GROUP_LABELS[case.group],
                    "model_key": case.key,
                    "model_label": case.label,
                    "source_trajectory": str(case.path),
                    "frames_analyzed": int(result["frames"]),
                    "keep_fraction": f"{KEEP_FRACTION:.2f}",
                    "bin_width_A": f"{DZ_BIN:.2f}",
                    "mean_dissociation_fraction": (
                        f"{float(result['dissociation_fraction']):.8f}"
                    ),
                    "mean_cos_theta_6_10A": (
                        f"{bulk_mean:.8f}" if np.isfinite(bulk_mean) else ""
                    ),
                    "molecule_samples_6_10A": bulk_samples,
                }
            )


def make_figure(results: dict[str, dict[str, np.ndarray | float | int]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7.2,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(4.15, 7.3),
        dpi=500,
        sharex=True,
        sharey=True,
    )

    for ax, panel_letter, (group, panel_label) in zip(axes, "abc", GROUPS):
        group_cases = [case for case in CASES if case.group == group]
        for case in group_cases:
            result = results[case.key]
            ax.plot(
                result["z"],
                result["orientation"],
                color=case.color,
                linestyle=case.linestyle,
                linewidth=2.0,
                alpha=0.95,
                label=case.label,
                zorder=3,
            )

        ax.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
        ax.set_xlim(0.6, 10.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel(r"$\langle\cos\theta\rangle$", labelpad=3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.tick_params(top=False, right=False)
        ax.text(
            -0.155,
            0.95,
            panel_letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontfamily="Arial Black",
            fontsize=13,
            clip_on=False,
        )
        ax.text(
            0.98,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
        )
        if group == "SCAN":
            legend_loc = "upper center"
            legend_anchor = (0.46, 0.99)
        else:
            legend_loc = "upper center"
            legend_anchor = (0.50, 0.99)
        legend = ax.legend(
            frameon=False,
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            ncol=1,
            handlelength=1.5,
            handletextpad=0.35,
            labelspacing=0.25,
            columnspacing=0.6,
        )
        for text, case in zip(legend.get_texts(), group_cases):
            if case.key in FINETUNED_CASE_KEYS:
                text.set_color(FINETUNED_COLOR)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(
        r"Distance from $\mathrm{TiO}_2$ surface [$\mathrm{\AA}$]",
        labelpad=3,
    )

    fig.subplots_adjust(
        left=0.17,
        right=0.98,
        bottom=0.08,
        top=0.96,
        hspace=0.07,
    )
    fig.savefig(OUTPUT_PNG, dpi=500)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {case.key: calculate_case(case) for case in CASES}
    write_csv(results)
    make_figure(results)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_SUMMARY_CSV}")
    for case in CASES:
        result = results[case.key]
        print(
            f"{case.label}: {result['frames']} frames, "
            f"mean dissociation={result['dissociation_fraction']:.2%}"
        )


if __name__ == "__main__":
    main()
