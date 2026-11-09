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

def make_figure() -> None:
    results = load_paired_profiles('foundation_mid_vs_caceles_large_cos_theta.dat', MODEL_GROUPS, DATA_STEMS, 'mean_cos_theta')

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

    fig.savefig(OUTPUT_PNG, dpi=500)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")

def main():
    global OUTPUT_PDF, OUTPUT_PNG
    directory = output_directory()
    OUTPUT_PDF = directory / 'Fig7_foundation_mid_vs_caceles_large_cos_theta.pdf'
    OUTPUT_PNG = directory / 'Fig7_foundation_mid_vs_caceles_large_cos_theta.png'
    make_figure()

if __name__ == "__main__":
    main()
