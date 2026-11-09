#!/usr/bin/env python3
"""Compare old and new large-system CACELES water-density profiles.

The three-row, half-A4-width layout follows
``plot_foundation_mid_vs_caceles_large_density.py``.  For each model family,
the 60-ps production result is shown as a black dashed line and the extended
100-ps production result as a model-colored solid line.  Density profiles are taken from the common
new/previous-large analysis, which uses the final 75% of saved XYZ frames and
averages the two TiO2-water interfaces.
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


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "small-Direct_MD_foundation"
ANALYSIS_DIR = SOURCE_ROOT / "Water-dipole-analyse"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "Sensitivity_MDduration"
    else ANALYSIS_DIR / "results" / "new_previous_large_caceles"
)
OUTPUT_PNG = OUTPUT_DIR / "old_new_large_water_density.png"
OUTPUT_PDF = OUTPUT_DIR / "old_new_large_water_density.pdf"
OUTPUT_DATA = OUTPUT_DIR / "old_new_large_water_density.dat"

sys.path.insert(0, str(ANALYSIS_DIR))

from compare_new_previous_large_caceles import MODELS, analyze  # noqa: E402


def write_profile_table(
    results: dict[tuple[str, str], dict[str, np.ndarray | float | int]],
) -> None:
    """Write every plotted curve using its native density grid."""
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for model in MODELS:
        old = results[(model.key, "previous")]
        new = results[(model.key, "new")]
        columns.extend(
            (
                np.asarray(old["density_z"]),
                np.asarray(old["density"]),
                np.asarray(new["density_z"]),
                np.asarray(new["density"]),
            )
        )
        labels.extend(
            (
                f"{model.key}_old_z_A",
                f"{model.key}_old_density_g_mL",
                f"{model.key}_new_z_A",
                f"{model.key}_new_density_g_mL",
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
            "60_ps_production; extended_100_ps_production; "
            "final_75_percent_of_saved_XYZ_frames\n"
            + " ".join(labels)
        ),
    )


def load_profile_table() -> dict[
    tuple[str, str], dict[str, np.ndarray]
]:
    """Load the saved profiles when the external trajectory disk is absent."""
    table = np.loadtxt(OUTPUT_DATA)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    results: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for model_index, model in enumerate(MODELS):
        first_column = 4 * model_index
        for version, offset in (("previous", 0), ("new", 2)):
            z = table[:, first_column + offset]
            density = table[:, first_column + offset + 1]
            valid = np.isfinite(z) & np.isfinite(density)
            results[(model.key, version)] = {
                "density_z": z[valid],
                "density": density[valid],
            }
    return results


def make_figure() -> None:
    try:
        results = {
            (model.key, version): analyze(model, version)
            for model in MODELS
            for version in ("previous", "new")
        }
    except FileNotFoundError as error:
        if not OUTPUT_DATA.is_file():
            raise
        print(
            f"Trajectory unavailable ({error}); loading {OUTPUT_DATA}",
            flush=True,
        )
        results = load_profile_table()

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

    # Match the reference layout: three stacked panels at half A4 width.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(105.0 / 25.4, 7.2),
        dpi=500,
        sharex=True,
        sharey=True,
    )

    for ax, panel_letter, model in zip(axes, "abc", MODELS):
        old = results[(model.key, "previous")]
        new = results[(model.key, "new")]
        ax.plot(
            old["density_z"],
            old["density"],
            color="0.10",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
        )
        ax.plot(
            new["density_z"],
            new["density"],
            color=model.color,
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
        )
        ax.set_xlim(0.6, 10.0)
        # PET-OMATPES reaches 4.58 g/mL, so 4.5 clips its first density peak.
        # Use one common 0-5 scale for all panels to preserve direct comparison.
        ax.set_ylim(0.0, 5.0)
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
            model.label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.0,
            fontweight="bold",
        )

    legend_handles = [
        Line2D([], [], color="0.10", linestyle="--", linewidth=2.0),
        Line2D([], [], color="0.25", linestyle="-", linewidth=2.0),
    ]
    fig.legend(
        legend_handles,
        ["60-ps\nProduction", "Extended 100-ps\nProduction"],
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
