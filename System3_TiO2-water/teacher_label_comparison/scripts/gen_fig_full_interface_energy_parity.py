"""Full-interface relative-energy parity from audited, geometry-matched labels.

One additive offset per model over the same full-interface subset. All points
are retained in the main axes; identical insets enlarge the dense energy range.
"""
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PACKAGE_DIR = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_DIR / "data"
OUT = PACKAGE_DIR / "reproduced"
STEM = "pbe_teacher_energy_parity"
MODELS = ["GemNet", "MACE_MH", "MACE_MP"]
COLORS = {"GemNet": "#D55E00", "MACE_MP": "#0072B2"}
NAMES = {"GemNet": "GemNet-OC22", "MACE_MH": "MACE-MH-1(OMat)", "MACE_MP": "MACE-MP-0(L)"}
FIGSIZE_IN = (147 / 25.4, 76 / 25.4)
FONT_SIZE_PT = 11


def parse_args(description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT,
                        help="Input cache directory (default: package data/).")
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="Output directory (default: package reproduced/).")
    return parser.parse_args()


def load_energies(data_dir=ROOT, output_dir=OUT):
    """Recompute centering from cached energies and check the previous export."""
    with np.load(data_dir / "cache_SCAN.npz", allow_pickle=False) as cache:
        mask = cache["n"] == 426
    frames = np.flatnonzero(mask)
    assert len(frames) == 3168
    energies, means = {}, {}
    table = pd.DataFrame({"frame": frames, "n_atoms": 426})
    with np.load(data_dir / "pairing.npz", allow_pickle=False) as pairing:
        for name in MODELS:
            with np.load(data_dir / f"cache_{name}.npz", allow_pickle=False) as cache:
                order = pairing[name]
                assert np.all(cache["n"][order][mask] == 426)
                energy = cache["energy"][order][mask].astype(float)
            means[name] = float(energy.mean())
            energies[name] = (energy - means[name]) * 1000 / 426
            assert np.all(np.isfinite(energies[name]))
            table[name + "_relative_energy_meV_atom"] = energies[name]
    old = pd.read_csv(data_dir / "environment_analysis" / "centered_energies.csv").set_index("frame").loc[frames]
    for name in MODELS:
        np.testing.assert_allclose(
            energies[name], old[name + "_centered_meV_atom"], rtol=0, atol=1e-8
        )
    table.to_csv(output_dir / f"{STEM}_data.csv", index=False)
    return energies, means


def style_panel(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.85)
    ax.tick_params(which="major", direction="in", length=3.5, width=0.8,
                   top=True, right=True, pad=4)
    ax.set_aspect("equal", adjustable="box")


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT, "axes.titlesize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT, "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
    })
    energy, means = load_energies(args.data_dir, output_dir)
    x = energy["MACE_MH"]
    compare = ["GemNet", "MACE_MP"]
    limits = (-60, 90)
    zoom_limits = (-20, 10)
    metrics = {}
    for name in compare:
        y = energy[name]
        d = y - x
        assert x.min() > limits[0] and x.max() < limits[1]
        assert y.min() > limits[0] and y.max() < limits[1]
        metrics[name] = {
            "rmsd_meV_atom": float(np.sqrt(np.mean(d * d))),
            "mae_meV_atom": float(np.mean(np.abs(d))),
            "mean_difference_meV_atom": float(d.mean()),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "points_in_main_panel": len(x),
            "points_in_zoom": int(np.sum((x >= -20) & (x <= 10) & (y >= -20) & (y <= 10))),
        }
    ratio = metrics["GemNet"]["rmsd_meV_atom"] / metrics["MACE_MP"]["rmsd_meV_atom"]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_IN)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.24, wspace=0.50)

    for idx, (ax, name) in enumerate(zip(axes, compare)):
        y = energy[name]
        style_panel(ax)
        ax.scatter(x, y, s=2.45, color=COLORS[name], alpha=0.48,
                   linewidths=0, edgecolors="none", zorder=2)
        ax.plot(limits, limits, color="#686868", lw=0.8, zorder=3)
        ax.set(xlim=limits, ylim=limits)
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.yaxis.set_major_locator(MultipleLocator(30))
        ax.set_xlabel(f'{NAMES["MACE_MH"]}\nenergy [meV/atom]', labelpad=5)
        ax.set_ylabel(f"{NAMES[name]}\nenergy [meV/atom]", labelpad=6)
        ax.text(0.045, 0.945, 'RMSD', transform=ax.transAxes,
                va="top", fontsize=FONT_SIZE_PT, color="black")
        ax.text(0.045, 0.865, f'{metrics[name]["rmsd_meV_atom"]:.2f} meV/atom',
                transform=ax.transAxes, va="top", fontsize=FONT_SIZE_PT,
                color="black", fontweight="bold")

        # Lower-right insets lie wholly away from the data in the main panels.
        inset_bounds = [0.61, 0.09, 0.35, 0.35]
        inset_x = [limits[0] + q * (limits[1] - limits[0]) for q in (0.61, 0.96)]
        inset_y = [limits[0] + q * (limits[1] - limits[0]) for q in (0.09, 0.44)]
        assert not np.any((x >= inset_x[0]) & (x <= inset_x[1]) &
                          (y >= inset_y[0]) & (y <= inset_y[1]))
        iax = ax.inset_axes(inset_bounds)
        style_panel(iax)
        iax.scatter(x, y, s=1.176, color=COLORS[name], alpha=0.42,
                    linewidths=0, edgecolors="none", zorder=2)
        iax.plot(zoom_limits, zoom_limits, color="#686868", lw=0.7, zorder=3)
        iax.set(xlim=zoom_limits, ylim=zoom_limits,
                 xticks=[-20, 10], yticks=[-20, 10])
        iax.tick_params(labelsize=FONT_SIZE_PT, length=2.3, pad=2)

    from matplotlib.text import Text
    assert all(abs(t.get_fontsize() - 11) < 1e-9 for t in fig.findobj(Text) if t.get_text())

    for ext in ["png"]:
        # Preserve exactly 70% of A4 paper width; tight bounding-box cropping changes it.
        fig.savefig(output_dir / f"{STEM}.{ext}", dpi=400, bbox_inches=None)
    plt.close(fig)

    metadata = {
        "reference": "MACE-MH-1(OMat)",
        "reference_role": "Common comparison baseline, not ground truth",
        "metric_definition": "RMSD = root-mean-square difference between model labels",
        "figure_size_mm": [147, 76], "font_size_pt": FONT_SIZE_PT,
        "rmsd_text_color": "black",
        "n_configurations": len(x), "n_atoms_per_configuration": 426,
        "alignment": "Audited exact-geometry mapping from pairing.npz",
        "relative_energy_definition": "1000 * (E_model - mean(E_model over the 3168 configurations)) / 426 [meV/atom]",
        "model_mean_total_energy_eV": means,
        "main_axis_limits_meV_atom": list(limits),
        "inset_axis_limits_meV_atom": list(zoom_limits),
        "normalization": "One additive energy offset per model; no scaling or linear fitting",
        "metrics": metrics, "gemnet_to_mace_mp_rmsd_ratio": ratio,
    }
    (output_dir / f"{STEM}_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    caption = (
        "Figure S9. Comparison of energy labels from GemNet-OC22 (left) and "
        "MACE-MP-0(L) (right) with MACE-MH-1(OMat) for the same 3,168 configurations of the "
        "anatase TiO₂(101)–water interface, each containing 426 atoms. Each point "
        "represents one configuration. For each model, the mean total energy over "
        "these configurations is subtracted before division by the number of atoms. "
        "Gray solid lines denote equal predictions, and both insets use the same "
        "magnified axis ranges. RMSD denotes the root-mean-square difference between "
        "the plotted energies, with MACE-MH-1(OMat) serving as the common comparison "
        "baseline.\n"
    )
    (output_dir / f"{STEM}_caption.txt").write_text(caption)
    print(json.dumps(metadata, indent=2))
    print("Created", output_dir / STEM)


if __name__ == "__main__":
    main()
