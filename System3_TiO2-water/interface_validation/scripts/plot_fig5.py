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

@dataclass(frozen=True)
class Case:
    key: str
    relative_path: str
    label: str
    group: str
    color: str
    linestyle: object
    legacy_cache_name: str | None = None

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
FINETUNED_CASE_KEYS = {"mace_mp_ft10", "mace_mp_ft50"}
FINETUNED_COLOR = "#D62728"

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

def main():
    global OUTPUT_PDF, OUTPUT_PNG
    directory = output_directory()
    OUTPUT_PDF = directory / 'Fig5_student_water_orientation.pdf'
    OUTPUT_PNG = directory / 'Fig5_student_water_orientation.png'
    make_figure(load_orientation_csv())

if __name__ == "__main__":
    main()
