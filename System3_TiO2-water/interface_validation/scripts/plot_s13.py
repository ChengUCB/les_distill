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

DENSITY_DATA = DATA_DIR / "caceles_mid_vs_large_density.dat"
ORIENTATION_DATA = DATA_DIR / "caceles_mid_vs_large_cos_theta.dat"
MODEL_GROUPS = [
    ("GemNet-OC22", "GemNet_OC22", "GemNet_OC22", "#B68FFE"),
    ("MACE-MH-1(Omat)", "MACE_MH_Omat", "MACE_MH_Omat", "#B1DFF2"),
    ("MACE-MP-0(L)", "MACE_MP0", "MACE_MP_0", "#5E6CB2"),
]

def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.6, 10.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.tick_params(top=False, right=False)

def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17 if label in {"a", "d"} else -0.08,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontfamily="Arial Black",
        fontsize=12,
        clip_on=False,
    )

def add_model_title(ax: plt.Axes, title: str) -> None:
    """Place the model name clearly inside the top of each panel."""
    ax.text(
        0.5,
        0.93,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )

def make_figure() -> None:
    density = load_named_table(DENSITY_DATA)
    orientation = load_named_table(ORIENTATION_DATA)

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
        2,
        3,
        figsize=(180.0 / 25.4, 116.0 / 25.4),
        dpi=500,
        sharex=True,
        sharey="row",
    )

    for column, (title, density_stem, orientation_stem, color) in enumerate(
        MODEL_GROUPS
    ):
        density_ax = axes[0, column]
        density_ax.plot(
            density[f"{density_stem}_mid_CACELES_z_A"],
            density[f"{density_stem}_mid_CACELES_density_g_mL"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            zorder=3,
        )
        density_ax.plot(
            density[f"{density_stem}_large_CACELES_z_A"],
            density[f"{density_stem}_large_CACELES_density_g_mL"],
            color="0.10",
            linestyle="--",
            linewidth=2.0,
            zorder=2,
        )
        density_ax.set_ylim(0.0, 4.5)
        density_ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        density_ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        add_model_title(density_ax, title)
        style_axis(density_ax)
        add_panel_label(density_ax, "abc"[column])

        orientation_ax = axes[1, column]
        orientation_ax.plot(
            orientation[f"{orientation_stem}_mid_CACELES_z_A"],
            orientation[f"{orientation_stem}_mid_CACELES_mean_cos_theta"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            zorder=3,
        )
        orientation_ax.plot(
            orientation[f"{orientation_stem}_large_CACELES_z_A"],
            orientation[f"{orientation_stem}_large_CACELES_mean_cos_theta"],
            color="0.10",
            linestyle="--",
            linewidth=2.0,
            zorder=2,
        )
        orientation_ax.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
        orientation_ax.set_ylim(-1.0, 1.0)
        orientation_ax.yaxis.set_major_locator(
            ticker.FixedLocator([-1.0, 0.0, 1.0])
        )
        orientation_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        orientation_ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        style_axis(orientation_ax)
        add_model_title(orientation_ax, title)
        add_panel_label(orientation_ax, "def"[column])

    axes[0, 0].set_ylabel("Water density [g/mL]", labelpad=4)
    axes[1, 0].set_ylabel(r"$\langle\cos\theta\rangle$", labelpad=4)

    legend_handles = [
        Line2D([], [], color="0.25", linestyle="-", linewidth=2.0),
        Line2D([], [], color="0.10", linestyle="--", linewidth=2.0),
    ]
    fig.legend(
        legend_handles,
        ["CACELES student (1,422 atoms)", "CACELES student (8,748 atoms)"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=2,
        handlelength=2.6,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    fig.supxlabel(
        r"Distance from $\mathrm{TiO}_2$ surface [$\mathrm{\AA}$]",
        x=0.53,
        y=0.018,
        fontsize=10.0,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.99,
        bottom=0.125,
        top=0.88,
        wspace=0.13,
        hspace=0.16,
    )

    fig.savefig(OUTPUT_PNG, dpi=500)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")

def main():
    global OUTPUT_PDF, OUTPUT_PNG
    directory = output_directory()
    OUTPUT_PDF = directory / 'caceles_mid_vs_large_system_size.pdf'
    OUTPUT_PNG = directory / 'caceles_mid_vs_large_system_size.png'
    make_figure()

if __name__ == "__main__":
    main()
