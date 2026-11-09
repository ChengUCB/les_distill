#!/usr/bin/env python3
"""MD inference-time figure, DPA4 / LAMBench Fig. 3 layout.

Layout conventions taken from that figure:
  - single boxed panel, all four spines, ticks pointing inward on all sides
  - log-log, x = number of atoms
  - no gridlines
  - one legend entry per model
  - a curve simply stops at the last size that fit in GPU memory

Figure geometry: the canvas is exactly half the width of an A4 page (105 mm) at
11 pt type (10 pt in the legend), so it drops into a two-column layout at 1:1
scale. At that width the
reference figure's legend-outside-right does not fit 13 series, so the legend sits
above the panel in two columns. Saved without a tight bbox, so the output width is
exactly 105 mm rather than whatever trimming would leave.

Identity encoding:
  - hue = model family
  - inside a family, every member differs in lightness AND line style AND marker,
    so no two curves of one family are told apart by colour alone
  - MACELES, the student model of this work, is highlighted in saturated red
  - the two eSEN curves use a coordinated violet pair so they remain distinct
    from the highlighted student model

y quantity: the mean wall time of one molecular-dynamics step, measured as the
timed block of 100 steps divided by 100. Lower is faster. Each step is dominated
by the model's energy and force evaluation on a new configuration but also
contains the integrator update and neighbour-list construction, so the honest
name is "wall time per MD step", not "inference time". The unit is a pure time
[ms] because "per MD step" is already carried by the quantity's name; labelling a
"time" in [ms step^-1] would give a time the dimensions of a rate.

Every calculator is shown at one precision only: MACE-LES is the fp32 build, and
every other calculator runs its own checkpoint/API default.

Reads speed_tests_combined.csv written by analyze_speed_tests.py.
Writes fig11_inference_time.png and .pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "speed_tests_combined.csv")
ok = df[(df["status"] == "ok") & (df["model"] != "maceles-orbmol-cons-fp64")].copy()

A4_WIDTH_MM = 210.0
FIG_W_IN = (A4_WIDTH_MM / 2) / 25.4      # 105 mm
FIG_H_IN = 4.55
FS = 11                                   # axis labels and tick labels
FS_LEGEND = 10                             # legend runs one point smaller

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": FS, "axes.labelsize": FS,
    "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "legend.fontsize": FS_LEGEND, "legend.frameon": False,
    "figure.dpi": 130, "savefig.dpi": 600,
    "lines.linewidth": 1.3,
})

# model -> (colour, marker, linestyle, legend label)
# grouped by family; inside a family the lightness, the line style and the marker
# all change together. This work first.
# Labels name the training task/dataset of each checkpoint, as requested.
# GemNet-OC22 keeps its name: the task (OC22) is already in it.
SERIES = [
    ("maceles-orbmol-cons-fp32",        "#d62728", "o", "-",  "MACELES student"),
    ("orb-v3-direct-omol",              "#f5c142", "o", "-",  "Orb-v3-OrbMol-d"),
    ("orb-v3-conservative-omol",        "#b07600", "s", "--", "Orb-v3-OrbMol"),
    ("uma-s-omol",                      "#86b6ef", "o", "-",  "UMA-S(Omol)"),
    ("uma-m-omol",                      "#1c5cab", "s", "--", "UMA-M(Omol)"),
    ("esen-md-dir",                     "#b39ddb", "o", "-",  "eSEN-OC25-md"),
    ("esen-sm-con",                     "#6a51a3", "s", "--", "eSEN-OC25-sm"),
    ("mace-off24-medium",               "#7fd9b6", "o", "-",  "MACE-OFF24(M)"),
    ("mace-mh-1",                       "#2fc78d", "s", "--", "MACE-MH-1(OMol)"),
    ("mace-mp0",                        "#17916a", "^", "-.", "MACE-MP-0(L)"),
    ("mace-omol-xl",                    "#0a5239", "D", ":",  "MACE-OMol"),
    ("gemnet-oc22",                     "#e87ba4", "o", "-",  "GemNet-OC22"),
    # singleton family, given the neutral dark slot the reference figure uses for its
    # one stand-alone baseline
    ("upet-pet-omatpes-l-conservative", "#3a3a3a", "o", "-",  "PET-OMATPES"),
]

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=True)

for model, colour, marker, ls, label in SERIES:
    s = ok[ok["model"] == model].sort_values("atoms")
    if s.empty:
        continue
    student = "LES" in label
    ax.plot(s["atoms"], s["ms_step"], color=colour, marker=marker, linestyle=ls,
            markersize=4.4 if student else (3.2 if marker == "D" else 3.6),
            linewidth=2.1 if student else 1.3,      # the student curve reads first
            label=label, markeredgewidth=0, zorder=4 if student else 3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(9.0e1, 1.8e4)   # left edge below 100 so the 10^2 decade tick is labelled
ax.set_ylim(6.0, 3.0e4)
ax.set_xlabel(r"Number of atoms, $N$")
# "per MD step" belongs in the name, so the unit is a pure time. Writing
# "time [ms step^-1]" would name a time but give it the dimensions of a rate.
ax.set_ylabel(r"Wall time per MD step [ms]")
ax.grid(False)

# legend above the panel, two columns: at 105 mm there is no room beside the axes
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
          labelspacing=0.3, columnspacing=1.0, handlelength=2.4,
          handletextpad=0.5, borderaxespad=0.0)

# no tight bbox, so the canvas width is exactly half of A4
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig11_inference_time.{ext}")
plt.close(fig)
print(f"canvas: {FIG_W_IN * 25.4:.1f} x {FIG_H_IN * 25.4:.1f} mm "
      f"(A4 width = {A4_WIDTH_MM:.0f} mm), type size {FS} pt")

# ---- numbers for the caption ----
print(f"{'model':22s} {'ms/step @1152':>14s} {'ms/step @12288':>15s} {'largest N':>10s}")
for model, _, _, _, label in SERIES:
    s = ok[ok["model"] == model].sort_values("atoms").set_index("atoms")
    if s.empty:
        continue
    g = lambda n: f"{s.at[n, 'ms_step']:14,.1f}" if n in s.index else " " * 14
    print(f"{label:22s} {g(1152)} {g(12288):>15s} {int(s.index.max()):10,d}")
