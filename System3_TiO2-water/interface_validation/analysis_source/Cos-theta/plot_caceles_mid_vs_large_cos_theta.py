#!/usr/bin/env python3
"""Compare CACELES water-orientation profiles at two system sizes.

The three panels show students distilled from GemNet-OC22, MACE-MH-1(OMat),
and MACE-MP-0(L). The two trajectories contain 1,422 and 8,748 atoms. Profiles
are the equal arithmetic mean of the two interface-resolved profiles, using
the local inward surface normal at each interface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
MID_DATA = SCRIPT_DIR / "foundation_mid_vs_caceles_mid_cos_theta.dat"
LARGE_DATA = SCRIPT_DIR / "foundation_mid_vs_caceles_large_cos_theta.dat"
OUTPUT_PNG = SCRIPT_DIR / "caceles_mid_vs_large_cos_theta.png"
OUTPUT_PDF = SCRIPT_DIR / "caceles_mid_vs_large_cos_theta.pdf"
OUTPUT_DATA = SCRIPT_DIR / "caceles_mid_vs_large_cos_theta.dat"

MODEL_GROUPS = [
    ("GemNet-OC22", "GemNet_OC22", "#8f53fe"),
    ("MACE-MH-1(Omat)", "MACE_MH_Omat", "#87CEEB"),
    ("MACE-MP-0(L)", "MACE_MP_0", "#071d88"),
]


def load_named_table(path: Path) -> dict[str, np.ndarray]:
    """Read the column labels and numeric columns from an exported profile."""
    header_lines: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header_lines.append(line[1:].strip())
    if len(header_lines) < 2:
        raise ValueError(f"Missing column-name header in {path}")
    labels = header_lines[-1].split()
    values = np.loadtxt(path)
    if values.shape[1] != len(labels):
        raise ValueError(
            f"{path}: {values.shape[1]} data columns but {len(labels)} labels"
        )
    return {label: values[:, index] for index, label in enumerate(labels)}


def write_profile_table(
    mid: dict[str, np.ndarray],
    large: dict[str, np.ndarray],
) -> None:
    """Write all six plotted profiles to one reproducible text table."""
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for _, stem, _ in MODEL_GROUPS:
        mid_z = mid[f"{stem}_mid_CACELES_z_A"]
        mid_profile = mid[f"{stem}_mid_CACELES_mean_cos_theta"]
        large_z = large[f"{stem}_large_CACELES_z_A"]
        large_profile = large[f"{stem}_large_CACELES_mean_cos_theta"]
        columns.extend((mid_z, mid_profile, large_z, large_profile))
        labels.extend(
            (
                f"{stem}_mid_CACELES_z_A",
                f"{stem}_mid_CACELES_mean_cos_theta",
                f"{stem}_large_CACELES_z_A",
                f"{stem}_large_CACELES_mean_cos_theta",
            )
        )
    np.savetxt(
        OUTPUT_DATA,
        np.column_stack(columns),
        fmt="%.8f",
        header=(
            "local_inward_normal; equal_arithmetic_mean_of_lower_upper_interfaces\n"
            + " ".join(labels)
        ),
    )


def make_figure() -> None:
    mid = load_named_table(MID_DATA)
    large = load_named_table(LARGE_DATA)

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

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(105.0 / 25.4, 7.2),
        dpi=500,
        sharex=True,
        sharey=True,
    )

    for ax, panel_letter, (title, stem, color) in zip(
        axes, "abc", MODEL_GROUPS
    ):
        ax.plot(
            mid[f"{stem}_mid_CACELES_z_A"],
            mid[f"{stem}_mid_CACELES_mean_cos_theta"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
        )
        ax.plot(
            large[f"{stem}_large_CACELES_z_A"],
            large[f"{stem}_large_CACELES_mean_cos_theta"],
            color="0.10",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
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

    write_profile_table(mid, large)
    fig.savefig(OUTPUT_PNG, dpi=500)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_DATA}")


if __name__ == "__main__":
    make_figure()
