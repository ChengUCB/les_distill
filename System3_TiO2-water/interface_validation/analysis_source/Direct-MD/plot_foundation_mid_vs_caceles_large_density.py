#!/usr/bin/env python3
"""Compare mid-cell foundation and large-cell CACELES water densities.

The three panels correspond to the GemNet-OC22, MACE-MH-1(Omat), and
MACE-MP-0(L) model families.  Processing follows
``plot_fig1_false_metallization_mid.py``: the final 75% trajectory caches are
used and density is normalized by the actual xy area without amplitude
rescaling.  Water oxygens follow the established density-pipeline Ti-neighbor
classification; this intentionally differs from the intact-H2O-only dipole
analysis.  Because different cell sizes are compared, the horizontal
coordinate is the distance from the nearest TiO2 surface.  The left and right
interfaces are calculated separately and then averaged.
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
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1] / "small-Direct_MD_foundation"
ANALYSIS_DIR = ROOT / "Water-dipole-analyse"
DENSITY_DIR = ROOT / "water_density_analysis"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "Direct-MD"
    else ANALYSIS_DIR / "results"
)
OUTPUT_PNG = OUTPUT_DIR / "foundation_mid_vs_caceles_large_density.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig6_foundation_mid_vs_caceles_large_density.pdf"
OUTPUT_DATA = OUTPUT_DIR / "foundation_mid_vs_caceles_large_density.dat"

sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(DENSITY_DIR))

import analyze_dft_system_size_normalization as core  # noqa: E402
from analyze_water_dipole import load_full  # noqa: E402


# Panel title, family color, mid foundation cache, large CACELES cache, and
# their surface-plane tolerances from the established density analysis.
MODEL_GROUPS = [
    ("GemNet-OC22", "#8f53fe", "GemNet_foundation_mid", "GemNet_large", 0.74, 0.80),
    ("MACE-MH-1(Omat)", "#87CEEB", "MH_foundation_mid", "MH_large", 0.74, 0.80),
    ("MACE-MP-0(L)", "#071d88", "MP_foundation_mid", "MP_large", 0.74, 0.80),
]

DATA_STEMS = {
    "GemNet-OC22": "GemNet_OC22",
    "MACE-MH-1(Omat)": "MACE_MH_Omat",
    "MACE-MP-0(L)": "MACE_MP0",
}


def load_density(name: str, surface_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the standard-pipeline mean of left/right interface densities."""
    positions, symbols, box = load_full(name)
    cell = np.diag(box)
    slab_idx, water_o_idx = core.classify_atoms(
        symbols, positions[0].astype(float), cell
    )
    area = float(box[0] * box[1])
    dz = core.BIN_WIDTH
    edges = np.arange(core.DZ_MIN, core.DZ_MAX + 0.5 * dz, dz)
    left_counts = np.zeros(len(edges) - 1)
    right_counts = np.zeros(len(edges) - 1)

    for frame in positions:
        z_frame = frame[:, 2].astype(float)
        slab_z = z_frame[slab_idx]
        water_z = z_frame[water_o_idx]
        bottom, top, _, _ = core.surface_planes(
            slab_z, "legacy_tol", surface_tolerance
        )
        midpoint = 0.5 * (bottom + top)
        left_water = water_z[water_z <= midpoint] - bottom
        right_water = top - water_z[water_z > midpoint]
        left_counts += np.histogram(left_water, bins=edges)[0]
        right_counts += np.histogram(right_water, bins=edges)[0]

    conversion = (
        1.0
        / len(positions)
        / (area * dz)
        * 18.01528
        / 6.02214076e23
        * 1.0e24
    )
    left_density = left_counts * conversion
    right_density = right_counts * conversion
    raw_density = 0.5 * (left_density + right_density)
    density = savgol_filter(raw_density, window_length=7, polyorder=3)
    return 0.5 * (edges[:-1] + edges[1:]), density


def write_profile_table(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Write all plotted curves to a single reproducible text table."""
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for title, _, foundation_name, student_name, _, _ in MODEL_GROUPS:
        z_foundation, density_foundation = results[foundation_name]
        z_student, density_student = results[student_name]
        columns.extend((z_foundation, density_foundation, z_student, density_student))
        stem = DATA_STEMS[title]
        labels.extend(
            (
                f"{stem}_mid_foundation_z_A",
                f"{stem}_mid_foundation_density_g_mL",
                f"{stem}_large_CACELES_z_A",
                f"{stem}_large_CACELES_density_g_mL",
            )
        )
    # Mid and large cells have different z dimensions and therefore different
    # grid lengths.  Keep their native coordinates and pad only the table.
    nrows = max(len(column) for column in columns)
    table = np.full((nrows, len(columns)), np.nan)
    for index, column in enumerate(columns):
        table[: len(column), index] = column
    np.savetxt(
        OUTPUT_DATA,
        table,
        fmt="%.8f",
        header="left_right_interface_arithmetic_mean\n" + " ".join(labels),
    )


def make_figure() -> None:
    cases = [
        (foundation, foundation_tol)
        for _, _, foundation, _, foundation_tol, _ in MODEL_GROUPS
    ] + [
        (student, student_tol)
        for _, _, _, student, _, student_tol in MODEL_GROUPS
    ]
    results = {name: load_density(name, tolerance) for name, tolerance in cases}

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

    # Three stacked panels at half the width of an A4 page (105 mm).
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
        z_foundation, density_foundation = results[foundation_name]
        z_student, density_student = results[student_name]

        ax.plot(
            z_foundation,
            density_foundation,
            color=color,
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
            label="Foundation model",
            zorder=3,
        )
        ax.plot(
            z_student,
            density_student,
            color="0.10",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label="CACELES student",
            zorder=2,
        )
        ax.set_xlim(0.6, 10.0)
        ax.set_ylim(0.0, 4.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(top=False, right=False)
        ax.set_ylabel("Density [g/mL]", labelpad=3)
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
        ["Foundation model\n(1,422 atoms)", "CACELES student\n(8,748 atoms)"],
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
