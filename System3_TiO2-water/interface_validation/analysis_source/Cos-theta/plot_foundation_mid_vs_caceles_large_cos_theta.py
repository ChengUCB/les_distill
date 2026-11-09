#!/usr/bin/env python3
"""Compare mid-cell foundation and large-cell CACELES water orientations.

The model grouping, colors, system pairing, typography, and stacked 1x3
layout follow ``plot_foundation_mid_vs_caceles_large_density.py``.  Water
orientation is the O-to-H-midpoint direction projected onto the local inward
interface normal: +z at the lower interface and -z at the upper interface.
The two interface profiles are calculated independently and then averaged
with equal weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1] / "small-Direct_MD_foundation"
ANALYSIS_DIR = ROOT / "Water-dipole-analyse"
DENSITY_DIR = ROOT / "water_density_analysis"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "Cos-theta"
    else ANALYSIS_DIR / "results"
)
OUTPUT_PNG = OUTPUT_DIR / "foundation_mid_vs_caceles_large_cos_theta.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig7_foundation_mid_vs_caceles_large_cos_theta.pdf"
OUTPUT_DATA = OUTPUT_DIR / "foundation_mid_vs_caceles_large_cos_theta.dat"

sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(DENSITY_DIR))

import analyze_dft_system_size_normalization as core  # noqa: E402
from analyze_water_dipole import load_full  # noqa: E402


# Panel label, family color, mid foundation cache, large CACELES cache, and
# surface-plane tolerances from the established density/orientation analysis.
MODEL_GROUPS = [
    ("GemNet-OC22", "#8f53fe", "GemNet_foundation_mid", "GemNet_large", 0.74, 0.80),
    ("MACE-MH-1(Omat)", "#87CEEB", "MH_foundation_mid", "MH_large", 0.74, 0.80),
    ("MACE-MP-0(L)", "#071d88", "MP_foundation_mid", "MP_large", 0.74, 0.80),
]

DATA_STEMS = {
    "GemNet-OC22": "GemNet_OC22",
    "MACE-MH-1(Omat)": "MACE_MH_Omat",
    "MACE-MP-0(L)": "MACE_MP_0",
}

LEGEND_LABELS = [
    "Foundation model\n(1,422 atoms)",
    "CACELES student\n(8,748 atoms)",
]

DZ_MIN = 0.0
DZ_MAX = 12.0
DZ_BIN = 0.10
MIN_INTERFACE_COUNT = 25


def _interface_mean(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Return a per-interface mean, masking bins with insufficient samples."""
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts >= MIN_INTERFACE_COUNT,
    )


def load_orientation(
    name: str,
    surface_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-weight lower/upper local-normal orientation profiles."""
    positions, symbols, box = load_full(name)
    cell = np.diag(box)
    slab_idx, water_o_idx = core.classify_atoms(
        symbols,
        positions[0].astype(float),
        cell,
    )
    o_idx = np.flatnonzero(symbols == "O")
    h_idx = np.flatnonzero(symbols == "H")
    is_water_atom = np.zeros(len(symbols), dtype=bool)
    is_water_atom[water_o_idx] = True
    is_water_o = is_water_atom[o_idx]

    edges = np.arange(DZ_MIN, DZ_MAX + 0.5 * DZ_BIN, DZ_BIN)
    nbins = len(edges) - 1
    lower_sum = np.zeros(nbins)
    lower_count = np.zeros(nbins)
    upper_sum = np.zeros(nbins)
    upper_count = np.zeros(nbins)

    for frame in positions:
        p = frame.astype(float)
        bottom, top, _, _ = core.surface_planes(
            p[slab_idx, 2],
            "legacy_tol",
            surface_tolerance,
        )
        midpoint = 0.5 * (bottom + top)

        # Reconstruct intact water molecules every frame using periodic
        # nearest-O ownership, so atom order and wrapped molecules are safe.
        tree = cKDTree(np.mod(p[o_idx], box), boxsize=box)
        _, owner = tree.query(np.mod(p[h_idx], box))
        owner_count = np.bincount(owner, minlength=len(o_idx))
        intact = np.flatnonzero(is_water_o & (owner_count == 2))

        order = np.argsort(owner, kind="stable")
        sorted_owner = owner[order]
        first = np.searchsorted(sorted_owner, intact, side="left")
        oxygen = p[o_idx[intact]]
        d1 = p[h_idx[order[first]]] - oxygen
        d2 = p[h_idx[order[first + 1]]] - oxygen
        d1 -= box * np.round(d1 / box)
        d2 -= box * np.round(d2 / box)
        bisector = 0.5 * (d1 + d2)
        norm = np.linalg.norm(bisector, axis=1)
        valid_norm = norm > 1.0e-12

        lower = (oxygen[:, 2] <= midpoint) & valid_norm
        upper = (oxygen[:, 2] > midpoint) & valid_norm
        lower_dz = oxygen[lower, 2] - bottom
        upper_dz = top - oxygen[upper, 2]
        lower_cos = bisector[lower, 2] / norm[lower]
        upper_cos = -bisector[upper, 2] / norm[upper]

        lower_sum += np.histogram(
            lower_dz,
            bins=edges,
            weights=lower_cos,
        )[0]
        lower_count += np.histogram(lower_dz, bins=edges)[0]
        upper_sum += np.histogram(
            upper_dz,
            bins=edges,
            weights=upper_cos,
        )[0]
        upper_count += np.histogram(upper_dz, bins=edges)[0]

    lower_profile = _interface_mean(lower_sum, lower_count)
    upper_profile = _interface_mean(upper_sum, upper_count)
    profile = 0.5 * (lower_profile + upper_profile)
    z = 0.5 * (edges[:-1] + edges[1:])
    return z, profile, lower_profile, upper_profile


def write_profile_table(
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Write the plotted equal-interface-weight profiles."""
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for title, _, foundation_name, student_name, _, _ in MODEL_GROUPS:
        z_foundation, cos_foundation, _, _ = results[foundation_name]
        z_student, cos_student, _, _ = results[student_name]
        columns.extend((z_foundation, cos_foundation, z_student, cos_student))
        stem = DATA_STEMS[title]
        labels.extend(
            (
                f"{stem}_mid_foundation_z_A",
                f"{stem}_mid_foundation_mean_cos_theta",
                f"{stem}_large_CACELES_z_A",
                f"{stem}_large_CACELES_mean_cos_theta",
            )
        )
    nrows = max(len(column) for column in columns)
    table = np.full((nrows, len(columns)), np.nan)
    for index, column in enumerate(columns):
        table[: len(column), index] = column
    np.savetxt(
        OUTPUT_DATA,
        table,
        fmt="%.8f",
        header=(
            "local_inward_normal; equal_arithmetic_mean_of_lower_upper_interfaces\n"
            + " ".join(labels)
        ),
    )


def make_figure() -> None:
    cases = [
        (foundation, foundation_tol)
        for _, _, foundation, _, foundation_tol, _ in MODEL_GROUPS
    ] + [
        (student, student_tol)
        for _, _, _, student, _, student_tol in MODEL_GROUPS
    ]
    results = {
        name: load_orientation(name, tolerance)
        for name, tolerance in cases
    }

    plt.rcParams.update(
        {
            "font.size": 12.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.labelsize": 11.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 11.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Match the density comparison: three stacked panels at half A4 width.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(105.0 / 25.4, 7.2),
        dpi=500,
        sharex=True,
        sharey=True,
    )

    for ax, panel_letter, (title, color, foundation_name, student_name, _, _) in zip(
        axes,
        "abc",
        MODEL_GROUPS,
    ):
        z_foundation, cos_foundation, _, _ = results[foundation_name]
        z_student, cos_student, _, _ = results[student_name]
        ax.plot(
            z_foundation,
            cos_foundation,
            color=color,
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
            label="Foundation model",
            zorder=3,
        )
        ax.plot(
            z_student,
            cos_student,
            color="0.10",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label="CACELES student",
            zorder=2,
        )
        ax.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
        ax.set_xlim(0.6, 10.0)
        ax.set_ylim(-1.0, 1.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.tick_params(top=False, right=False)
        ax.set_ylabel(r"$\langle\cos\theta\rangle$", labelpad=3)
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
            0.5,
            0.965,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.0,
            fontweight="bold",
        )

    legend_handles = [
        Line2D([], [], color="0.25", linestyle="-", linewidth=2.0),
        Line2D([], [], color="0.10", linestyle="--", linewidth=2.0),
    ]
    fig.legend(
        legend_handles,
        LEGEND_LABELS,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.975),
        ncol=2,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.supxlabel(
        r"Distance from $\mathrm{TiO}_2$ surface [$\mathrm{\AA}$]",
        x=0.57,
        y=0.012,
        fontsize=11.0,
    )
    fig.subplots_adjust(
        left=0.19,
        right=0.98,
        bottom=0.08,
        top=0.89,
        hspace=0.10,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_profile_table(results)
    fig.savefig(OUTPUT_PNG, dpi=500)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_DATA}")


if __name__ == "__main__":
    make_figure()
