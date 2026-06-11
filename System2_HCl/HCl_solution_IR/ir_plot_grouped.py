# -*- coding: utf-8 -*-
"""
IR spectra of 2M HCl solution grouped by DFT functional.
2-panel figure: GGA (top) | Hybrid/ωB97M-V (bottom).

Data layout (relative to this script):
  data/IR_spectra/<keyword>.npz   — pre-computed H3O⁺ IR spectra
  data/reference/Exp_2M_HCl_2D_IR.csv
  data/reference/water_IR.pkl
"""
import os
import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 500,
    }
)

# ─────────────────────────── paths ───────────────────────────
_THIS_DIR = Path(__file__).parent
IR_SPECTRA_DIR = _THIS_DIR / "data" / "IR_spectra"
EXP_DATA_PATH  = _THIS_DIR / "data" / "reference" / "Exp_2M_HCl_2D_IR.csv"
EXP_BULK_PATH  = _THIS_DIR / "data" / "reference" / "water_IR.pkl"
OUTPUT_PLOT    = str(_THIS_DIR / "IR_Comparison_Grouped_By_DFT.png")

# ─────────────────────────── constants ───────────────────────────
FIG_W, FIG_H = 6.2, 7.0
PEAK_SEARCH_RANGE = (2800, 3800)

COLOR_POOL = [
    "#d83034", "#008dff", "#4ecb8d", "#7971ea",
    "#f78ef0", "#c701ff", "#a98467", "#333333",
]

THEORY_STYLE_MAP = {
    "RPBE-D3": "--",
    "RPBE": "-.",
    "$\\omega$B97M-V": "-",
    "PBE": ":",
    "default": "-",
}

HYBRID_MODEL_STYLE_MAP = {
    "MACE-MH-1(OMol)": "--",
    "UMA-M(OMol)": "-",
    "Orb-v3-OrbMol": "-.",
    "MACE-OMol": ":",
}

MODEL_COLOR_OVERRIDE = {
    "MACE-MP-0(L)": "#f28705",
    "UMA-M(OC20)": "#ff69b4",
    "eSEN-OC25-sm": "#c35dc3",
    "eSEN-OC25-md": "#8f56b3",
    "MACE-OMol": "#a98467",
}

# (keyword, display_name, functional)  — keyword matches IR_spectra/<keyword>.npz
MODEL_CONFIGS = [
    ("UMA-S-omol",       "UMA-S(OMol)",       "$\\omega$B97M-V"),
    ("MACE_MH",          "MACE-MH-1(OMol)",   "$\\omega$B97M-V"),
    ("MACE_omol",        "MACE-OMol",         "$\\omega$B97M-V"),
    ("temp_orbmol",      "Orb-v3-OrbMol",     "$\\omega$B97M-V"),
    ("esen-oc25-sm-con", "eSEN-OC25-sm",      "RPBE-D3"),
    ("esen-oc25-md-dir", "eSEN-OC25-md",      "RPBE-D3"),
    ("UMA-M-OC20",       "UMA-M(OC20)",       "RPBE"),
    ("MACE_MP0",         "MACE-MP-0(L)",      "PBE"),
]

GROUP_DEFS = [
    {
        "panel_label": "GGA",
        "legend_title": "GGA",
        "theories": {"PBE", "RPBE", "RPBE-D3"},
    },
    {
        "panel_label": "Hybrid",
        "legend_title": "Hybrid ($\\omega$B97M-V)",
        "theories": {"$\\omega$B97M-V"},
    },
]

EXP_LABELS = {"Bulk water", "Aqueous proton"}

PEAK_ANNOTS = [
    (1150, "Zundel Proton\nStretch"),
    (1850, "Protonated water\nBend"),
    (2400, "Eigen-like\nOH Stretch"),
    (3000, "Zundel-like\nOH Stretch"),
]


def normalize_by_peak(wn, inten, xlim):
    mask = (wn >= xlim[0]) & (wn <= xlim[1])
    if not np.any(mask):
        return None
    peak_mask = (wn >= PEAK_SEARCH_RANGE[0]) & (wn <= PEAK_SEARCH_RANGE[1])
    curr_peak = np.max(inten[peak_mask]) if np.any(peak_mask) else np.max(inten[mask])
    return inten / curr_peak if curr_peak > 1e-12 else inten


def draw_background(ax, xlim):
    if EXP_BULK_PATH.exists():
        try:
            with open(EXP_BULK_PATH, "rb") as f:
                exp_data = pickle.load(f)
            wn_bulk, int_bulk = exp_data["Exp"][0], exp_data["Exp"][1]
            ax.fill_between(
                wn_bulk, int_bulk / np.max(int_bulk),
                color="#666666", alpha=0.25, edgecolor="none",
                label="Bulk water", zorder=1,
            )
        except Exception:
            pass

    if EXP_DATA_PATH.exists():
        try:
            exp = np.genfromtxt(str(EXP_DATA_PATH), delimiter=",")
            exp_wn, exp_int = exp[:, 0], exp[:, 1]
            exp_int = exp_int - np.min(exp_int)
            e_mask = (exp_wn > 3000) & (exp_wn <= 3800)
            if np.any(e_mask):
                exp_int = exp_int / np.max(exp_int[e_mask])
                p_mask = (exp_wn >= xlim[0]) & (exp_wn <= xlim[1])
                ax.plot(
                    exp_wn[p_mask], exp_int[p_mask],
                    color="black", ls="--", lw=1.2,
                    label="Aqueous proton", zorder=3,
                )
        except Exception:
            pass


def load_spectra(xlim):
    spectra = []
    color_cycler = itertools.cycle(COLOR_POOL)
    for keyword, display_name, functional in MODEL_CONFIGS:
        npz_path = IR_SPECTRA_DIR / f"{keyword}.npz"
        if not npz_path.exists():
            print(f" [skip] missing {npz_path.name}")
            continue
        d = np.load(npz_path, allow_pickle=True)
        wn, inten = d["omega"], d["intensity"]
        i_norm = normalize_by_peak(wn, inten, xlim)
        if i_norm is None:
            continue
        spectra.append({
            "name":      display_name,
            "functional": functional,
            "wn":        wn,
            "inten":     i_norm,
            "color":     MODEL_COLOR_OVERRIDE.get(display_name, next(color_cycler)),
            "linestyle": HYBRID_MODEL_STYLE_MAP.get(
                display_name,
                THEORY_STYLE_MAP.get(functional, THEORY_STYLE_MAP["default"]),
            ),
        })
    return spectra


def main():
    xlim = (1000, 4000)
    spectra = load_spectra(xlim)

    fig, axs = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), dpi=500, sharex=True)
    group_legends = []
    exp_legend_map = {}

    for ax, group_cfg in zip(axs, GROUP_DEFS):
        group_name   = group_cfg["panel_label"]
        legend_title = group_cfg["legend_title"]
        group_theories = group_cfg["theories"]
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.grid(False)
        draw_background(ax, xlim)

        for spec in spectra:
            if spec["functional"] not in group_theories:
                continue
            mask = (spec["wn"] >= xlim[0]) & (spec["wn"] <= xlim[1])
            if not np.any(mask):
                continue
            ax.plot(
                spec["wn"][mask], spec["inten"][mask],
                color=spec["color"], linestyle=spec["linestyle"],
                linewidth=1.8,
                label=(
                    spec["name"] if "$\\omega$B97M-V" in group_theories
                    else f"{spec['name']} ({spec['functional']})"
                ),
                alpha=0.9, zorder=2,
            )

        ax.axvspan(2000, 3200, facecolor="#d0f0d4", edgecolor="none", alpha=0.4, zorder=0)
        ax.text(
            0.98, 0.97, group_name,
            fontsize=11, fontweight="bold", ha="right", va="top",
            transform=ax.transAxes, color="#333333",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
        )
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.4 if group_name == "Hybrid" else 1.7)
        ax.set_ylabel("Intensity [a.u.]", labelpad=5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))
        ax.tick_params(top=True, bottom=True, right=True, left=True, direction="in")
        if group_name == "GGA":
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            paired = list(zip(handles, labels))
            exp_items   = [p for p in paired if p[1] in EXP_LABELS]
            model_items = [p for p in paired if p[1] not in EXP_LABELS]
            if group_name == "Hybrid":
                model_items = sorted(model_items, key=lambda p: (0 if p[1].startswith("UMA-M(OMol)") else 1))
            elif group_name == "GGA":
                model_items = sorted(model_items, key=lambda p: (0 if p[1].startswith("MACE-MP-0(L)") else 1))
            group_legends.append((legend_title, group_name, [p[0] for p in model_items], [p[1] for p in model_items]))
            for handle, label in exp_items:
                if label not in exp_legend_map:
                    exp_legend_map[label] = handle

    EIGEN_ANNOT = (2400, "Eigen-like\nOH Stretch")
    for x_v, txt in PEAK_ANNOTS:
        if (x_v, txt) != EIGEN_ANNOT:
            for ax in axs:
                ax.axvline(x_v, color="#aaaaaa", linestyle="--", linewidth=0.9, zorder=0, clip_on=True)

    legend_y = 0.995
    legend_x_positions = {"GGA": 0.25, "Exp": 0.55, "Hybrid": 0.85}
    for legend_title, group_name, handles, labels in group_legends:
        legend_x = legend_x_positions["GGA"] if group_name == "GGA" else legend_x_positions["Hybrid"]
        fig.legend(
            handles, labels, title=legend_title, frameon=False,
            loc="upper center", bbox_to_anchor=(legend_x, legend_y),
            ncol=1, fontsize=8, title_fontsize=9,
            handlelength=2.2, handletextpad=0.3, columnspacing=0.6, labelspacing=0.2,
        )

    if exp_legend_map:
        exp_labels  = list(exp_legend_map.keys())
        exp_handles = [exp_legend_map[l] for l in exp_labels]
        fig.legend(
            exp_handles, exp_labels, title="Exp", frameon=False,
            loc="upper center", bbox_to_anchor=(legend_x_positions["Exp"], legend_y),
            ncol=1, fontsize=8, title_fontsize=9,
            handlelength=2.2, handletextpad=0.3, columnspacing=0.6, labelspacing=0.2,
        )

    axs[-1].set_xlabel(r"Wavenumber [cm$^{-1}$]", labelpad=5)
    plt.tight_layout(rect=(0, 0, 1, 0.9), h_pad=3.0)

    pos0, pos1 = axs[0].get_position(), axs[1].get_position()
    mid_y = (pos0.y0 + pos1.y1) / 2.0
    from matplotlib.transforms import blended_transform_factory
    mid_trans = blended_transform_factory(axs[0].transData, fig.transFigure)
    for x_v, txt in PEAK_ANNOTS:
        fig.text(x_v, mid_y, txt, fontsize=8.2, ha="center", va="center",
                 transform=mid_trans, color="black", zorder=5)
    fig.text(2700, mid_y, "+", fontsize=11, fontweight="bold", ha="center", va="center",
             transform=mid_trans, color="black", zorder=5)

    plt.savefig(OUTPUT_PLOT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
