#!/usr/bin/env python3
"""Compare mid-cell and large-cell CACELES water densities.

The three stacked panels correspond to GemNet-OC22, MACE-MH-1(Omat), and
MACE-MP-0(L).  The mid and large CACELES trajectories contain 1,422 and 8,748
atoms, respectively.  Density is computed with the same left/right-interface
averaging and normalization used by
``plot_foundation_mid_vs_caceles_large_density.py``.
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


ROOT = Path(__file__).resolve().parents[1] / "small-Direct_MD_foundation"
ANALYSIS_DIR = ROOT / "Water-dipole-analyse"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "Direct-MD"
    else ANALYSIS_DIR / "results"
)
OUTPUT_PNG = OUTPUT_DIR / "caceles_mid_vs_large_density.png"
OUTPUT_PDF = OUTPUT_DIR / "caceles_mid_vs_large_density.pdf"
OUTPUT_DATA = OUTPUT_DIR / "caceles_mid_vs_large_density.dat"

sys.path.insert(0, str(SCRIPT_DIR))

from plot_foundation_mid_vs_caceles_large_density import load_density  # noqa: E402


# Panel title, mid-CACELES cache, large-CACELES cache, and surface tolerances.
MODEL_GROUPS = [
    ("GemNet-OC22", "GemNet_mid", "GemNet_large", 0.74, 0.80),
    ("MACE-MH-1(Omat)", "MH_mid", "MH_large", 0.74, 0.80),
    ("MACE-MP-0(L)", "MP_mid", "MP_large", 0.74, 0.80),
]

DATA_STEMS = {
    "GemNet-OC22": "GemNet_OC22",
    "MACE-MH-1(Omat)": "MACE_MH_Omat",
    "MACE-MP-0(L)": "MACE_MP0",
}

# Match the model-family colors in the foundation-versus-student figure.
MODEL_COLORS = {
    "GemNet-OC22": "#B68FFE",
    "MACE-MH-1(Omat)": "#B1DFF2",
    "MACE-MP-0(L)": "#5E6CB2",
}
LARGE_COLOR = "#000000"
MID_LINESTYLE = "-"
LARGE_LINESTYLE = "--"


def write_profile_table(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Write all plotted profiles to a reproducible text table."""
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for title, mid_name, large_name, _, _ in MODEL_GROUPS:
        z_mid, density_mid = results[mid_name]
        z_large, density_large = results[large_name]
        columns.extend((z_mid, density_mid, z_large, density_large))
        stem = DATA_STEMS[title]
        labels.extend(
            (
                f"{stem}_mid_CACELES_z_A",
                f"{stem}_mid_CACELES_density_g_mL",
                f"{stem}_large_CACELES_z_A",
                f"{stem}_large_CACELES_density_g_mL",
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
        header="left_right_interface_arithmetic_mean\n" + " ".join(labels),
    )


def make_figure() -> None:
    cases = [
        (mid_name, mid_tolerance)
        for _, mid_name, _, mid_tolerance, _ in MODEL_GROUPS
    ] + [
        (large_name, large_tolerance)
        for _, _, large_name, _, large_tolerance in MODEL_GROUPS
    ]
    results = {
        name: load_density(name, tolerance)
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

    # Three stacked panels at half the width of an A4 page (105 mm).
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(105.0 / 25.4, 7.2),
        dpi=500,
        sharex=True,
        sharey=True,
    )

    for ax, panel_letter, (title, mid_name, large_name, _, _) in zip(
        axes,
        "abc",
        MODEL_GROUPS,
    ):
        z_mid, density_mid = results[mid_name]
        z_large, density_large = results[large_name]

        ax.plot(
            z_mid,
            density_mid,
            color=MODEL_COLORS[title],
            linestyle=MID_LINESTYLE,
            linewidth=2.0,
            alpha=1.0,
            zorder=3,
        )
        ax.plot(
            z_large,
            density_large,
            color=LARGE_COLOR,
            linestyle=LARGE_LINESTYLE,
            linewidth=2.0,
            alpha=1.0,
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
            fontsize=11,
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
        Line2D(
            [], [], color="0.25",
            linestyle=MID_LINESTYLE, linewidth=2.0,
        ),
        Line2D(
            [], [], color=LARGE_COLOR,
            linestyle=LARGE_LINESTYLE, linewidth=2.0,
        ),
    ]
    fig.legend(
        legend_handles,
        ["CACELES student\n(1,422 atoms)", "CACELES student\n(8,748 atoms)"],
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
