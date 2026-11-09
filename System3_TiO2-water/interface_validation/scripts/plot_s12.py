#!/usr/bin/env python3
"""Redraw the final archived curves. Original analysis is preserved separately.

The plotting instructions are retained from the original source; only data
loading and output paths are adapted. No binning, smoothing or averaging here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from plot_support import DATA_DIR, output_directory, load_named_table, load_orientation_csv, load_paired_profiles

INPUT_DATA = DATA_DIR / "old_new_large_water_density.dat"
MODELS = [
    SimpleNamespace(key="mace_mp0", label=r"MACE-MP-0(L)", color="#071d88"),
    SimpleNamespace(key="uma_oc20", label=r"UMA-M(OC20)", color="#D81B60"),
    SimpleNamespace(key="pet_omatpes", label=r"PET-OMATPES (r$^2$SCAN)", color="#E76F51"),
]

def load_profile_table() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load the profiles exported by the original vertical figure."""
    table = np.loadtxt(INPUT_DATA)
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
    results = load_profile_table()

    plt.rcParams.update(
        {
            "font.size": 10.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 10.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 / 25.4, 78.0 / 25.4),
        dpi=500,
        sharex=True,
        sharey=True,
    )

    for ax, panel_letter, model in zip(axes, "abc", MODELS):
        previous = results[(model.key, "previous")]
        extended = results[(model.key, "new")]
        ax.plot(
            previous["density_z"],
            previous["density"],
            color="0.10",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
        )
        ax.plot(
            extended["density_z"],
            extended["density"],
            color=model.color,
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
        )
        ax.set_xlim(0.6, 10.0)
        ax.set_ylim(0.0, 5.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(top=False, right=False)
        ax.text(
            -0.17 if panel_letter == "a" else -0.08,
            1.02,
            panel_letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontfamily="Arial Black",
            fontsize=12,
            clip_on=False,
        )
        ax.text(
            0.5,
            0.965,
            model.label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )

    axes[0].set_ylabel("Water density [g/mL]", labelpad=4)

    legend_handles = [
        Line2D([], [], color="0.10", linestyle="--", linewidth=2.0),
        Line2D([], [], color="0.25", linestyle="-", linewidth=2.0),
    ]
    fig.legend(
        legend_handles,
        ["60-ps production", "Extended 100-ps production"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.94),
        ncol=2,
        handlelength=2.6,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    fig.supxlabel(
        r"Distance from $\mathrm{TiO}_2$ surface [$\mathrm{\AA}$]",
        x=0.53,
        y=0.025,
        fontsize=10.0,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.99,
        bottom=0.23,
        top=0.76,
        wspace=0.13,
    )

    fig.savefig(OUTPUT_PNG, dpi=500)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")

def main():
    global OUTPUT_PDF, OUTPUT_PNG
    directory = output_directory()
    OUTPUT_PDF = directory / 'extended_water_density.pdf'
    OUTPUT_PNG = directory / 'extended_water_density.png'
    make_figure()

if __name__ == "__main__":
    main()
