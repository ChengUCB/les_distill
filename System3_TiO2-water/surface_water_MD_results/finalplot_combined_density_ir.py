#!/usr/bin/env python3
"""
Combined figure: water density profiles (left column) + IR spectra (right column).
3 × 2 layout — rows share the same functional group: PBE | RPBE(-D3) | SCAN.
A compact table legend (interfacial vs bulk line-style key) sits above the right
column only; the left column top is white space, keeping all row baselines aligned.

Layout (GridSpec 4 × 2, height_ratios = [legend_frac, 1, 1, 1]):
  gs[0,0]  ← empty           gs[0,1]  ← table legend
  gs[1,0]  ← density PBE     gs[1,1]  ← IR PBE
  gs[2,0]  ← density RPBE    gs[2,1]  ← IR RPBE(-D3)
  gs[3,0]  ← density SCAN    gs[3,1]  ← IR SCAN

Data layout (relative to this script):
  data/density/MD_simulations/<model>/TiO2-water-density_profile.txt
  data/reference/JCP_TiO2_water_den_profile.csv
  data/reference/JCP_DPLR_water_at_interface_fig2.csv
  data/reference/JCP_EXP_water_at_interface_fig2.csv
  data/reference/water_IR.pkl
  IR_spectra/<model_key>_<layer>.npz   ← pre-computed (run merge_pkl.py to regenerate)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────── global style ───────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "dejavusans",
    }
)

# ─────────────────────────── paths ───────────────────────────
_THIS_DIR = Path(__file__).parent

# density
DENSITY_DIR      = _THIS_DIR / "data" / "water_density_profile"
DENSITY_WORK_DIR = DENSITY_DIR / "MD_simulations"
REF_CSV_PATH    = _THIS_DIR / "data" / "reference" / "JCP_TiO2_water_den_profile.csv"
DENSITY_DATA_FILENAME = "TiO2-water-density_profile.txt"

# IR
EXP_BULK_PATH = _THIS_DIR / "data" / "reference" / "water_IR.pkl"
EXP_INT_PATH  = _THIS_DIR / "data" / "reference" / "JCP_EXP_water_at_interface_fig2.csv"
DPLR_INT_PATH = _THIS_DIR / "data" / "reference" / "JCP_DPLR_water_at_interface_fig2.csv"
IR_SPECTRA_DIR = _THIS_DIR / "data" / "IR_spectra"

OUTPUT_FILE = _THIS_DIR / "TiO2_density_IR_combined.pdf"

# ─────────────────────────── shared constants ───────────────────────────
FIG_DPI = 500
LEGEND_FONT_SIZE = 7.2
TABLE_FONT_SIZE = LEGEND_FONT_SIZE
AXIS_LABEL_FONT_SIZE = 10

GROUP_DEFS = [("PBE", {"PBE"}), ("RPBE", {"RPBE"}), ("SCAN", {"SCAN"})]
PANEL_ANNOTATION_LABELS = {"PBE": "PBE", "RPBE": "RPBE(-D3)", "SCAN": "meta-GGA"}

SCAN_LEGEND_ORDER = [
    r"MACE-MP-0(L) $\rightarrow$ 10% SCAN dataset",
    r"MACE-MP-0(L) $\rightarrow$ 50% SCAN dataset",
    r"MACE-MH-1(r$^2$SCAN)",
    r"PET-OMATPES (r$^2$SCAN)",
    r"CACELES (5.5 $\mathrm{\AA}$)",
    r"CACELES (6 $\mathrm{\AA}$, T=1)",
    "DPLR (Zhang et al., 2025)",
]

# ─────────────────────────── density config ───────────────────────────
DENSITY_MODEL_CONFIGS = [
    ("MACE_MP0a_L",  r"MACE-MP-0(L)",                              "#071d88", "PBE"),
    ("MACE-MH-omat", r"MACE-MH-1(OMat)",                          "#87CEEB", "PBE"),
    ("OC22",         r"GemNet-OC22",                               "#8f53fe", "PBE"),
    ("UMA-M-oc20",   r"UMA-M(OC20)",                               "#D81B60", "RPBE"),
    ("UMA-S-OC25",   r"UMA-S(OC25)$^{\mathbf{\dagger}}$",         "#bc5090", "RPBE"),
    ("esen-sm-con",  r"eSEN-OC25-sm$^{\mathbf{\dagger}}$",        "#FF8C00", "RPBE"),
    ("esen-direct",  r"eSEN-OC25-md$^{\mathbf{\dagger}}$",        "#fd8bf5", "RPBE"),
    ("DFT",          r"CACELES (5.5 $\mathrm{\AA}$)",              "#ffb81d", "SCAN"),
    ("DFT_cut6",     r"CACELES (6 $\mathrm{\AA}$, T=1)",          "#ff9a8f", "SCAN"),
]

EXTRA_SCAN_DENSITY = [
    (
        DENSITY_WORK_DIR / "Finetune" / "MP0a_L_fine-tune_Fonly_frac10",
        r"MACE-MP-0(L) $\rightarrow$ 10% SCAN dataset",
        "#1F9E89",
        "SCAN",
    ),
    (
        DENSITY_WORK_DIR / "Finetune" / "MP0a_L_fine-tune_Fonly_frac50",
        r"MACE-MP-0(L) $\rightarrow$ 50% SCAN dataset",
        "#73D055",
        "SCAN",
    ),
    (
        DENSITY_WORK_DIR / "MACE-H1-r2scan",
        r"MACE-MH-1(r$^2$SCAN)",
        "#1d65a6",
        "SCAN",
    ),
    (
        DENSITY_WORK_DIR / "PET-r2scan",
        r"PET-OMATPES (r$^2$SCAN)",
        "#E76F51",
        "SCAN",
    ),
]

SCAN_PANEL_DEN_COLORS: dict[str, str] = {
    r"CACELES (5.5 $\mathrm{\AA}$)":                   "#ffb81d",
    r"CACELES (6 $\mathrm{\AA}$, T=1)":                "#ff9a8f",
    r"MACE-MP-0(L) $\rightarrow$ 10% SCAN dataset":     "#1F9E89",
    r"MACE-MP-0(L) $\rightarrow$ 50% SCAN dataset":     "#73D055",
    r"MACE-MH-1(r$^2$SCAN)":                            "#1d65a6",
    r"PET-OMATPES (r$^2$SCAN)":                         "#E76F51",
}

ALIGN_DENSITY = 0.1
DENSITY_LS_CYCLE = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]

DENSITY_EXCLUDE_BY_GROUP: dict[str, set[str]] = {
    "RPBE": {r"UMA-S(OC25)$^{\mathbf{\dagger}}$"},
}

# ─────────────────────────── IR config ───────────────────────────
# (model_key, label, color, theory, linestyle)
# model_key matches the filename stem in IR_spectra/<key>_<layer>.npz
IR_MODEL_CONFIG_LIST = [
    ("MACE_MP0a_L",    r"MACE-MP-0(L)",                                   "#071d88", "PBE",  (0, (3, 1, 1, 1))),
    ("MACE-MH-omat",   r"MACE-MH-1(OMat)",                               "#87CEEB", "PBE",  (0, (4, 1.5))),
    ("OC22",           r"GemNet-OC22",                                     "#8f53fe", "PBE",  ":"),
    ("UMA-oc20",       r"UMA-M(OC20)",                                     "#D81B60", "RPBE", "-"),
    ("esen-cons",      r"eSEN-OC25-sm$^{\mathbf{\dagger}}$",              "#FF8C00", "RPBE", (0, (1, 1))),
    ("esen-direct",    r"eSEN-OC25-md$^{\mathbf{\dagger}}$",              "#fd8bf5", "RPBE", (0, (3, 1, 1, 1, 1, 1))),
    ("DFT",            r"CACELES (5.5 $\mathrm{\AA}$)",                   "#ffb81d", "SCAN", "--"),
    ("DFT_CUT6",       r"CACELES (6 $\mathrm{\AA}$, T=1)",               "#ff9a8f", "SCAN", "-."),
    ("frac10",         r"MACE-MP-0(L) $\rightarrow$ 10% SCAN dataset",   "#1F9E89", "SCAN", ":"),
    ("frac50",         r"MACE-MP-0(L) $\rightarrow$ 50% SCAN dataset",   "#73D055", "SCAN", (0, (3, 1, 1, 1))),
    ("MACE-H1-r2scan", r"MACE-MH-1(r$^2$SCAN)",                          "#1d65a6", "SCAN", (0, (4, 1.5))),
    ("PET-r2scan",     r"PET-OMATPES (r$^2$SCAN)",                        "#E76F51", "SCAN", (0, (1, 1))),
]

IR_LAYER_CONFIGS = {
    "1st":  {"linewidth": 1.9, "alpha": 0.95, "zorder": 3},
    "Bulk": {"linewidth": 1.2, "alpha": 0.55, "zorder": 2},
}

PEAK_SEARCH_RANGE = (2800, 3800)


# ═══════════════════════════════════════════════════════════════════
#  Density helpers
# ═══════════════════════════════════════════════════════════════════

def first_crossing_x(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    if len(x) < 2:
        return None
    idx = np.argsort(x)
    xs, ys = np.asarray(x)[idx], np.asarray(y)[idx]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 < threshold <= y1) or (y0 <= threshold < y1):
            if y1 == y0:
                return float(xs[i])
            return float(xs[i] + (threshold - y0) / (y1 - y0) * (xs[i + 1] - xs[i]))
    above = np.where(ys >= threshold)[0]
    return float(xs[above[0]]) if above.size > 0 else None


def load_md_profile(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 2]


def smooth_ref_density(y: np.ndarray) -> np.ndarray:
    try:
        return gaussian_filter1d(y, sigma=1.0)
    except Exception:
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        return np.convolve(y, kernel, mode="same")


def load_density_curves() -> tuple[list[dict], dict | None]:
    curves: list[dict] = []
    for folder, label, color, theory in DENSITY_MODEL_CONFIGS:
        fpath = DENSITY_WORK_DIR / folder / DENSITY_DATA_FILENAME
        if not fpath.exists():
            print(f"[density] missing {fpath}, skip.")
            continue
        try:
            x, y = load_md_profile(fpath)
            curves.append({"label": label, "theory": theory, "x": x, "y": y,
                           "color": color, "linewidth": 2.0})
        except Exception as e:
            print(f"[density] error {label}: {e}")

    for model_dir, label, color, theory in EXTRA_SCAN_DENSITY:
        fpath = model_dir / DENSITY_DATA_FILENAME
        if not fpath.exists():
            print(f"[density] missing {fpath}, skip.")
            continue
        try:
            x, y = load_md_profile(fpath)
            curves.append({"label": label, "theory": theory, "x": x, "y": y,
                           "color": color, "linewidth": 2.0})
        except Exception as e:
            print(f"[density] error {label}: {e}")

    if not curves:
        raise RuntimeError("No density curves loaded.")

    anchor = next(
        (c for c in curves if r"CACELES (5.5 $\mathrm{\AA}$)" in c["label"]),
        curves[0],
    )
    anchor_x0 = first_crossing_x(anchor["x"], anchor["y"], ALIGN_DENSITY) or 1.0
    for c in curves:
        x0 = first_crossing_x(c["x"], c["y"], ALIGN_DENSITY)
        c["x_aligned"] = c["x"] + (anchor_x0 - x0) if x0 is not None else c["x"]

    ref: dict | None = None
    if REF_CSV_PATH.exists():
        try:
            raw = np.genfromtxt(REF_CSV_PATH, delimiter=",")
            rx, ry = raw[:, 0], smooth_ref_density(raw[:, 1])
            rx0 = first_crossing_x(rx, ry, ALIGN_DENSITY)
            ref = {"x": rx + (anchor_x0 - rx0) if rx0 else rx, "y": ry}
        except Exception as e:
            print(f"[density] error reading reference: {e}")

    return curves, ref


# ═══════════════════════════════════════════════════════════════════
#  IR helpers
# ═══════════════════════════════════════════════════════════════════

def _normalize_to_peak(omega: np.ndarray, intensity: np.ndarray,
                       wn_range: tuple[int, int]) -> np.ndarray:
    mask = (omega >= wn_range[0]) & (omega <= wn_range[1])
    peak = float(np.max(intensity[mask])) if mask.sum() > 0 else float(np.max(intensity))
    return intensity * (1.0 / peak) if abs(peak) > 1e-12 else intensity


def _load_ir_csv(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, bool]:
    path = str(filepath)
    if not os.path.exists(path):
        return np.array([]), np.array([]), False
    for skip in (0, 1):
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=skip)
            if data.ndim == 2 and data.shape[1] >= 2:
                idx = np.argsort(data[:, 0])
                return data[idx, 0], data[idx, 1], True
        except Exception:
            continue
    return np.array([]), np.array([]), False


def load_ir_spectra() -> list[dict]:
    spectra: list[dict] = []
    for key, label, color, theory, linestyle in IR_MODEL_CONFIG_LIST:
        for layer_name, layer_cfg in IR_LAYER_CONFIGS.items():
            path = IR_SPECTRA_DIR / f"{key}_{layer_name}.npz"
            if not path.exists():
                if layer_name == "1st":
                    print(f"[IR] missing {path.name}, skip.")
                continue
            try:
                d = np.load(path)
                wn, raw_y = d["omega"], d["intensity"]
            except Exception as e:
                print(f"[IR] error loading {path.name}: {e}")
                continue
            y_norm = _normalize_to_peak(wn, raw_y, PEAK_SEARCH_RANGE)
            layer_ls = "-" if layer_name == "1st" else "--"
            spectra.append({
                "theory":    theory,
                "label":     label,
                "layer":     layer_name,
                "color":     color,
                "linestyle": layer_ls,
                "linewidth": layer_cfg["linewidth"],
                "alpha":     layer_cfg["alpha"],
                "zorder":    layer_cfg["zorder"],
                "wn":        wn,
                "y":         y_norm,
            })
    return spectra


# ═══════════════════════════════════════════════════════════════════
#  Table legend
# ═══════════════════════════════════════════════════════════════════

def draw_table_legend(ax_tbl: plt.Axes) -> None:
    ax_tbl.set_axis_off()
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)

    x_label_end = 0.22
    x_int_l,  x_int_r  = x_label_end, 0.61
    x_bulk_l, x_bulk_r = 0.61, 1.0
    y_header, y_md, y_exp = 0.84, 0.48, 0.16

    for x, txt in [
        ((x_int_l + x_int_r) / 2,   "Interfacial water"),
        ((x_bulk_l + x_bulk_r) / 2, "Bulk water"),
    ]:
        ax_tbl.text(x, y_header, txt, ha="center", va="center",
                    fontsize=TABLE_FONT_SIZE, fontweight="bold")

    for y, txt in [(y_md, "Sim"), (y_exp, "Exp")]:
        ax_tbl.text(x_label_end / 2, y, txt, ha="center", va="center",
                    fontsize=TABLE_FONT_SIZE, fontweight="bold")

    _LINE_PAD = 0.39
    _LINE_LEN = 0.22

    def _line(xl: float, xr: float, y: float, ls: str, lw: float,
              alpha: float = 1.0) -> None:
        cw = xr - xl
        x0 = xl + _LINE_PAD * cw
        x1 = x0 + _LINE_LEN * cw
        ax_tbl.plot([x0, x1], [y, y], color="black", linestyle=ls,
                    linewidth=lw, alpha=alpha, solid_capstyle="round")

    _line(x_int_l,  x_int_r,  y_md,  "-",  1.8, alpha=0.95)
    _line(x_bulk_l, x_bulk_r, y_md,  "--", 1.4, alpha=0.85)
    _line(x_int_l,  x_int_r,  y_exp, ":",  1.4)

    bw = x_bulk_r - x_bulk_l
    ax_tbl.add_patch(Rectangle(
        (x_bulk_l + _LINE_PAD * bw, y_exp - 0.11),
        _LINE_LEN * bw, 0.22,
        facecolor="#666666", alpha=0.15, edgecolor="none",
    ))


# ═══════════════════════════════════════════════════════════════════
#  Panel helpers
# ═══════════════════════════════════════════════════════════════════

def _style_label(label: str) -> str:
    return label.replace(
        r"$\rightarrow$",
        r"$\boldsymbol{\rightarrow}$",
    )


def _add_phase_label(ax: plt.Axes, group_name: str, use_data_coords: bool = False,
                     x_data: float = 9.75, y_data: float = 4.8) -> None:
    txt = PANEL_ANNOTATION_LABELS.get(group_name, group_name)
    if use_data_coords:
        ax.text(x_data, y_data, txt, ha="right", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))
    else:
        ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontweight="bold")


def _ordered_legend(ax: plt.Axes, group_name: str,
                    font_size: float = LEGEND_FONT_SIZE,
                    loc: str = "upper left",
                    bbox_to_anchor: tuple = (0.01, 0.995),
                    ncol: int = 1,
                    handlelength: float = 2.2) -> plt.legend:
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return None
    if group_name == "SCAN":
        lbl_to_h = {l: h for h, l in zip(handles, labels)}
        ordered = [(lbl_to_h[l], l) for l in SCAN_LEGEND_ORDER if l in lbl_to_h]
    else:
        non_exp = [(h, l) for h, l in zip(handles, labels) if "Exp" not in l]
        exp     = [(h, l) for h, l in zip(handles, labels) if "Exp" in l]
        ordered = non_exp + exp
    if not ordered:
        return None
    display_labels = [_style_label(l) for _, l in ordered]
    leg = ax.legend(
        [o[0] for o in ordered], display_labels,
        frameon=False, fontsize=font_size,
        loc=loc, bbox_to_anchor=bbox_to_anchor,
        ncol=ncol, handletextpad=0.35, handlelength=handlelength,
        labelspacing=0.25, columnspacing=0.6,
    )
    for txt in leg.get_texts():
        if r"\rightarrow" in txt.get_text():
            txt.set_color("red")
    return leg


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    density_curves, density_ref = load_density_curves()
    ir_spectra = load_ir_spectra()

    exp_bulk_wn = exp_bulk_y = np.array([])
    if EXP_BULK_PATH.exists():
        try:
            with open(EXP_BULK_PATH, "rb") as f:
                ed = pickle.load(f)
            rb = np.real(np.asarray(ed["Exp"][1]))
            exp_bulk_wn = np.real(np.asarray(ed["Exp"][0]))
            exp_bulk_y  = rb / np.max(rb)
        except Exception as e:
            print(f"[IR] exp bulk: {e}")

    exp_int_wn, exp_int_y, ok_int = _load_ir_csv(EXP_INT_PATH)
    dplr_wn,    dplr_y,    ok_dplr = _load_ir_csv(DPLR_INT_PATH)
    if ok_int:
        exp_int_y = _normalize_to_peak(exp_int_wn, exp_int_y, PEAK_SEARCH_RANGE)
    if ok_dplr:
        dplr_y = _normalize_to_peak(dplr_wn, dplr_y, PEAK_SEARCH_RANGE)

    fig = plt.figure(figsize=(8.27, 8.0), dpi=FIG_DPI)
    gs = GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[0.16, 1, 1, 1],
        hspace=0.07,
        wspace=0.22,
        left=0.09, right=0.99, bottom=0.07, top=0.97,
    )

    ax_tbl = fig.add_subplot(gs[0, 1])
    draw_table_legend(ax_tbl)

    ax_den: list[plt.Axes] = []
    for i in range(3):
        kw = dict(sharex=ax_den[0], sharey=ax_den[0]) if i > 0 else {}
        ax_den.append(fig.add_subplot(gs[i + 1, 0], **kw))

    ax_ir: list[plt.Axes] = []
    for i in range(3):
        kw = dict(sharex=ax_ir[0], sharey=ax_ir[0]) if i > 0 else {}
        ax_ir.append(fig.add_subplot(gs[i + 1, 1], **kw))

    for ax in ax_den[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in ax_ir[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax, (group_name, group_set) in zip(ax_den, GROUP_DEFS):
        exclude = DENSITY_EXCLUDE_BY_GROUP.get(group_name, set())
        group_curves = [
            c for c in density_curves
            if c["theory"] in group_set and c["label"] not in exclude
        ]
        for si, c in enumerate(group_curves):
            col = SCAN_PANEL_DEN_COLORS.get(c["label"], c["color"]) if group_name == "SCAN" else c["color"]
            ax.plot(c["x_aligned"], c["y"],
                    color=col,
                    linestyle=DENSITY_LS_CYCLE[si % len(DENSITY_LS_CYCLE)],
                    linewidth=c["linewidth"],
                    alpha=0.95, label=c["label"], zorder=3)

        if group_name == "SCAN" and density_ref is not None:
            dplr_ls = DENSITY_LS_CYCLE[len(group_curves) % len(DENSITY_LS_CYCLE)]
            ax.plot(density_ref["x"], density_ref["y"],
                    color="#7A7A7A", linestyle=dplr_ls, linewidth=1.4,
                    label="DPLR (Zhang et al., 2025)", alpha=0.98, zorder=2)

        ax.set_xlim(0.6, 10.0)
        ax.set_ylim(0.0, 5.0)
        ax.set_ylabel(r"Density [g/mL]", labelpad=3, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(top=False, right=False)
        _add_phase_label(ax, group_name, use_data_coords=True)
        legend_anchor = (0.43, 0.99) if group_name == "SCAN" else (0.5, 0.99)
        _ordered_legend(
            ax, group_name,
            loc="upper center", bbox_to_anchor=legend_anchor,
            ncol=1, handlelength=1.5,
        )

    axs_den_rpbe = ax_den[1]

    for ax, (group_name, group_set) in zip(ax_ir, GROUP_DEFS):
        if exp_bulk_wn.size > 0:
            ax.fill_between(exp_bulk_wn, exp_bulk_y,
                            color="#666666", alpha=0.15, edgecolor="none",
                            label="_nolegend_", zorder=1)
        if ok_int:
            ax.plot(exp_int_wn, exp_int_y,
                    color="black", linestyle=":", linewidth=1.2,
                    label="_nolegend_", zorder=3)
        if group_name == "SCAN" and ok_dplr:
            ax.plot(dplr_wn, dplr_y,
                    color="#7A7A7A", linestyle="-", linewidth=1.3,
                    label="DPLR (Zhang et al., 2025)", zorder=3)

        for spec in ir_spectra:
            if spec["theory"] not in group_set:
                continue
            ax.plot(spec["wn"], spec["y"],
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    linewidth=spec["linewidth"],
                    alpha=spec["alpha"],
                    label=spec["label"] if spec["layer"] == "1st" else "_nolegend_",
                    zorder=spec["zorder"])

        ax.set_xlim(1500, 4000)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))
        ax.tick_params(top=False, right=False)
        ax.set_ylabel("Intensity [a.u.]", labelpad=3, fontsize=AXIS_LABEL_FONT_SIZE)
        _add_phase_label(ax, group_name)
        _ordered_legend(ax, group_name)

    axs_ir_rpbe = ax_ir[1]

    _SCAN_SPINE_LW = 2.2
    for ax in (ax_den[2], ax_ir[2]):
        for spine in ax.spines.values():
            spine.set_linewidth(_SCAN_SPINE_LW)
        ax.tick_params(width=_SCAN_SPINE_LW * 0.7)

    ax_den[-1].set_xlabel(
        r"Distance from $\mathrm{TiO}_2$ surface [$\mathrm{\AA}$]",
        labelpad=3, fontsize=AXIS_LABEL_FONT_SIZE,
    )
    ax_ir[-1].set_xlabel(
        r"Wavenumber [cm$^{-1}$]", labelpad=3, fontsize=AXIS_LABEL_FONT_SIZE,
    )

    fig.canvas.draw()
    for ax in (axs_den_rpbe, axs_ir_rpbe):
        leg = ax.get_legend()
        if leg is None:
            continue
        bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
        ax.text(
            bbox.x0 + bbox.width / 2,
            bbox.y0 + 0.02,
            r"$^{\mathbf{\dagger}}$with D3 correction",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=LEGEND_FONT_SIZE, color="black",
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    fig.savefig(OUTPUT_FILE.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {OUTPUT_FILE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
