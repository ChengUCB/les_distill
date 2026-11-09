"""All Cartesian force components on matched full-interface configurations.

Companion to the energy parity figure. No sampling, mean subtraction, projection,
or fitted correction; full-range panels show every force component.
"""
import json
import numpy as np

from gen_fig_full_interface_energy_parity import ROOT, OUT, COLORS, NAMES, FIGSIZE_IN, FONT_SIZE_PT, style_panel, parse_args
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

STEM = "pbe_teacher_force_parity"
COMPARE = ["GemNet", "MACE_MP"]


def load_forces(data_dir=ROOT):
    with np.load(data_dir / "cache_SCAN.npz", allow_pickle=False) as cache:
        frames = np.flatnonzero(cache["n"] == 426)
        offsets = cache["offsets"]
        numbers = cache["z"]
        species = np.concatenate([numbers[offsets[i]:offsets[i + 1]] for i in frames])
    assert len(frames) == 3168
    forces = {}
    with np.load(data_dir / "pairing.npz", allow_pickle=False) as pairing:
        for name in ["MACE_MH"] + COMPARE:
            with np.load(data_dir / f"cache_{name}.npz", allow_pickle=False) as cache:
                indices = pairing[name][frames]
                assert np.all(cache["n"][indices] == 426)
                offsets = cache["offsets"]
                values = cache["force"]
                numbers = cache["z"]
                paired_species = np.concatenate([numbers[offsets[i]:offsets[i + 1]] for i in indices])
                np.testing.assert_array_equal(paired_species, species)
                forces[name] = np.concatenate([values[offsets[i]:offsets[i + 1]] for i in indices])
            assert forces[name].shape == (3168 * 426, 3)
            assert np.all(np.isfinite(forces[name]))
    return forces, species


def main():
    args = parse_args(__doc__)
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
    forces, species = load_forces(args.data_dir)
    x = forces["MACE_MH"].reshape(-1)
    limits = (-10, 10)
    zoom_limits = (-1, 1)
    metrics = {}
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_IN)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.24, wspace=0.50)

    for idx, (ax, name) in enumerate(zip(axes, COMPARE)):
        y = forces[name].reshape(-1)
        delta = y - x
        m = {
            "rmsd_meV_A": float(np.sqrt(np.mean(delta ** 2)) * 1000),
            "mae_meV_A": float(np.mean(np.abs(delta)) * 1000),
            "mean_difference_meV_A": float(delta.mean() * 1000),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "points_in_main_panel": len(x),
            "points_in_zoom": int(np.sum((np.abs(x) <= 1) & (np.abs(y) <= 1))),
        }
        metrics[name] = m
        assert x.min() > limits[0] and x.max() < limits[1]
        assert y.min() > limits[0] and y.max() < limits[1]
        style_panel(ax)
        ax.scatter(x, y, s=0.245, color=COLORS[name], alpha=0.15,
                   linewidths=0, edgecolors="none", rasterized=True, zorder=2)
        ax.plot(limits, limits, color="#686868", lw=0.8, zorder=3)
        ax.set(xlim=limits, ylim=limits)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel(f'{NAMES["MACE_MH"]}\nforce [eV/Å]', labelpad=7)
        ax.set_ylabel(f"{NAMES[name]}\nforce [eV/Å]", labelpad=6)
        ax.text(0.045, 0.945, 'RMSD', transform=ax.transAxes,
                va="top", fontsize=FONT_SIZE_PT, color="black")
        ax.text(0.045, 0.865, f'{m["rmsd_meV_A"]:.2f} meV/Å',
                transform=ax.transAxes, va="top", fontsize=FONT_SIZE_PT,
                color="black", fontweight="bold")

        # Keep inset entirely off the main-panel data cloud.
        assert not np.any((x >= 2.2) & (x <= 9.2) & (y >= -8.2) & (y <= -1.2))
        iax = ax.inset_axes([0.61, 0.09, 0.35, 0.35])
        style_panel(iax)
        zoom_mask = (np.abs(x) <= 1) & (np.abs(y) <= 1)
        iax.scatter(x[zoom_mask], y[zoom_mask], s=0.245, color=COLORS[name], alpha=0.15,
                    linewidths=0, edgecolors="none", rasterized=True, zorder=2)
        iax.plot(zoom_limits, zoom_limits, color="#686868", lw=0.7, zorder=3)
        iax.set(xlim=zoom_limits, ylim=zoom_limits, xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        iax.tick_params(labelsize=FONT_SIZE_PT, length=2.3, pad=2)
        print(name, json.dumps(m), flush=True)

    from matplotlib.text import Text
    assert all(abs(t.get_fontsize() - 11) < 1e-9 for t in fig.findobj(Text) if t.get_text())

    for ext in ["png"]:
        # Preserve exactly 70% of A4 paper width; tight bounding-box cropping changes it.
        fig.savefig(output_dir / f"{STEM}.{ext}", dpi=400, bbox_inches=None)
        print("Created", output_dir / f"{STEM}.{ext}", flush=True)
    plt.close(fig)
    metadata = {
        "reference": NAMES["MACE_MH"],
        "reference_role": "Common comparison baseline, not ground truth",
        "metric_definition": "RMSD = root-mean-square difference between model labels",
        "figure_size_mm": [147, 76], "font_size_pt": FONT_SIZE_PT,
        "rmsd_text_color": "black",
        "model_names": NAMES,
        "n_configurations": 3168,
        "n_atoms_per_configuration": 426,
        "n_atom_records": len(species),
        "n_components_per_panel": len(x),
        "element_atom_records": {symbol: int(np.sum(species == z)) for symbol, z in [("Ti", 22), ("O", 8), ("H", 1)]},
        "point_definition": "One signed Cartesian force component Fx, Fy, or Fz on the same atom in the same configuration",
        "alignment": "Audited exact-geometry mapping from pairing.npz; selected canonical frames with 426 atoms",
        "force_treatment": "Unmodified stored force labels; no centering, rescaling, net-force removal, or subsampling",
        "main_axis_limits_eV_A": list(limits),
        "inset_axis_limits_eV_A": list(zoom_limits),
        "rendering": "All force components plotted with identical marker size and opacity; 400 dpi PNG, embedded as an opaque RGB image in the final PDF",
        "metrics": metrics,
    }
    (output_dir / f"{STEM}_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    caption = (
        "Figure S10. Comparison of force labels from GemNet-OC22 (left) and "
        "MACE-MP-0(L) (right) with MACE-MH-1(OMat) for the same 3,168 configurations of the "
        "anatase TiO₂(101)–water interface, each containing 426 atoms. Each point "
        "represents one signed Cartesian force component on a matched atom. All "
        "x, y, and z components of all Ti, O, and H atoms are included, giving "
        "4,048,704 components per panel. Gray solid lines denote equal predictions, "
        "and both insets use the same magnified axis ranges. RMSD denotes the "
        "root-mean-square difference over all force components, with "
        "MACE-MH-1(OMat) serving as the common comparison baseline.\n"
    )
    (output_dir / f"{STEM}_caption.txt").write_text(caption)


if __name__ == "__main__":
    main()
