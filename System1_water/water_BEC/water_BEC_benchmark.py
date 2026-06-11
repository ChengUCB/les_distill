#!/usr/bin/env python3
"""Reproduce water BEC benchmark plots from packaged relative-path data."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "water_bec_benchmark_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from sklearn.decomposition import KernelPCA  # noqa: E402
from sklearn.metrics import pairwise_distances  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
REF_FILENAME = "h2o_bec.xyz"
EPSILON_R = 1.78


THEORY_COLORS = {
    "RPBE-D3": "#0047eb",
    "RPBE": "#00bfff",
    "PBE": "#56b6b4",
    "wB97M-D3": "#eb00ad",
    "wB97M-V": "#ff6361",
}


PARITY_MODELS = [
    ("bec_H2O-esen-cons-oc25.xyz", "eSEN-OC25-sm", "RPBE-D3"),
    ("bec_H2O-aqcat-MD-500.xyz", "AQCat25-ev2", "RPBE"),
    ("bec_H2O-MACE-MP0_L.xyz", "MACE-MP-0(L)", "PBE"),
    ("bec_H2O-MACE-OFF-M.xyz", "MACE-OFF23(M)", r"$\omega$B97M-D3(BJ)"),
    ("bec_H2O-MACEOFF24-M.xyz", "MACE-OFF24(M)", r"$\omega$B97M-D3(BJ)"),
    ("bec_UMA-M.xyz", "UMA-M(OMol)", r"$\omega$B97M-V"),
    ("bec_MACE-omol.xyz", "MACE-OMol", r"$\omega$B97M-V"),
    ("bec_MACE-MH.xyz", "MACE-MH-1(OMol)", r"$\omega$B97M-V"),
    ("bec_Orb-omol-con.xyz", "Orb-v3-OrbMol", r"$\omega$B97M-V"),
    ("bec_Orb-omol-direct.xyz", "Orb-v3-OrbMol-d", r"$\omega$B97M-V"),
]


KPCA_MODELS = [
    ("bec_H2O-esen-cons-oc25.xyz", "eSEN-OC25-sm", "RPBE-D3", "v"),
    ("bec_H2O-aqcat-MD-500.xyz", "AQCat25-ev2", "RPBE", ">"),
    ("bec_H2O-MACE-MP0_L.xyz", "MACE-MP-0(L)", "PBE", "^"),
    ("bec_H2O-MACE-OFF-S.xyz", "MACE-OFF23(S)", "wB97M-D3", "D"),
    ("bec_H2O-MACE-OFF-M.xyz", "MACE-OFF23(M)", "wB97M-D3", "s"),
    ("bec_H2O-MACEOFF24-M.xyz", "MACE-OFF24(M)", "wB97M-D3", "p"),
    ("bec_H2O-MACELES.xyz", "MACELES-OFF", "wB97M-D3", "P"),
    ("bec_UMA-M.xyz", "UMA-M(OMol)", "wB97M-V", "*"),
    ("bec_MACE-omol.xyz", "MACE-OMol", "wB97M-V", "h"),
    ("bec_MACE-MH.xyz", "MACE-MH-1(OMol)", "wB97M-V", "o"),
    ("bec_Orb-omol-con.xyz", "Orb-v3-OrbMol", "wB97M-V", "8"),
    ("bec_Orb-omol-direct.xyz", "Orb-v3-OrbMol-d", "wB97M-V", "X"),
]


MARKER_SIZES = {
    "o": 45,
    "s": 45,
    "^": 55,
    "v": 55,
    "D": 35,
    ">": 55,
    "p": 60,
    "P": 60,
    "*": 110,
    "h": 60,
    "8": 60,
    "X": 65,
}


@dataclass(frozen=True)
class ModelStyle:
    name: str
    color: str
    marker: str
    theory: str


class SparseKPCA:
    """Small replacement for asaplib.pca.SPARSE_KPCA used in the original notebook."""

    def __init__(self, n_components: int = 2, kernel: dict | None = None):
        self.n_components = n_components
        self.kernel_config = kernel or {}
        self.kpca = KernelPCA(n_components=n_components, kernel="precomputed")

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        kernel_type = "linear"
        gamma = 1.0
        degree = 1

        if self.kernel_config:
            first_key = list(self.kernel_config.keys())[0]
            params = self.kernel_config[first_key]
            kernel_type = params.get("type", "linear")
            gamma = params.get("gamma", 1.0)
            degree = params.get("d", 1)

        if kernel_type == "rbf":
            d2 = pairwise_distances(x, metric="euclidean", squared=True)
            kernel = np.exp(-gamma * d2)
        elif kernel_type == "polynomial":
            kernel = np.clip(x @ x.T, -1.0, 1.0)
            kernel = np.power(kernel, degree)
        else:
            kernel = x @ x.T

        return self.kpca.fit_transform(kernel)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
        }
    )


def all_required_files() -> list[str]:
    names = {REF_FILENAME}
    names.update(filename for filename, *_ in PARITY_MODELS)
    names.update(filename for filename, *_ in KPCA_MODELS)
    return sorted(names)


def validate_inputs(data_dir: Path) -> None:
    missing = [data_dir / filename for filename in all_required_files() if not (data_dir / filename).exists()]
    if missing:
        rel_paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required benchmark files:\n{rel_paths}")


def bec_array(atoms):
    if "BEC" in atoms.arrays:
        return atoms.get_array("BEC")
    if "MACE_BEC" in atoms.arrays:
        return atoms.get_array("MACE_BEC")
    raise KeyError("Atoms object has neither BEC nor MACE_BEC array")


def flatten_bec_frames(frames, scale: float = 1.0) -> np.ndarray:
    return np.concatenate([(bec_array(frame) * scale).flatten() for frame in frames])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(r2), float(rmse)


def plot_bec_parity(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_name: str = "bec_parity_2x5_clean_scaled.png",
) -> Path:
    configure_matplotlib()
    validate_inputs(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_frames = read(data_dir / REF_FILENAME, index=":")
    all_ref = flatten_bec_frames(ref_frames)
    atomic_numbers = [frame.numbers for frame in ref_frames]
    an_rep = np.repeat(np.concatenate(atomic_numbers), 9)

    element_order = [el for el in [8, 7, 6, 1] if el in np.unique(np.concatenate(atomic_numbers))]
    element_colors = {1: "white", 8: "red", 6: "black", 7: "blue"}
    diag_mask = np.isin(np.arange(all_ref.size) % 9, [0, 4, 8])
    offdiag_mask = ~diag_mask
    scale_factor = EPSILON_R**0.5

    theory_colors = {
        "RPBE-D3": "#0047eb",
        "RPBE": "#00bfff",
        "PBE": "#56b6b4",
        r"$\omega$B97M-D3(BJ)": "#eb00ad",
        r"$\omega$B97M-V": "#ff6361",
    }

    half_a4_width = 4.135
    scale_ratio = 1.33
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(half_a4_width * scale_ratio, 1.85 * scale_ratio),
        dpi=600,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.12)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(PARITY_MODELS):
            ax.axis("off")
            continue

        filename, label_title, label_level = PARITY_MODELS[i]
        spine_color = theory_colors.get(label_level, "black")
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
            spine.set_linewidth(1.5)
        ax.tick_params(axis="both", colors=spine_color)
        ax.set_box_aspect(1)

        pred_frames = read(data_dir / filename, index=":")
        all_pred = flatten_bec_frames(pred_frames, scale=scale_factor)
        if all_ref.size != all_pred.size:
            raise ValueError(f"{filename} has {all_pred.size} BEC values, expected {all_ref.size}")

        r2, rmse = regression_metrics(all_ref, all_pred)
        final_pred = all_pred
        if r2 < 0.5:
            inv_r2, inv_rmse = regression_metrics(all_ref, -all_pred)
            if inv_r2 > r2:
                final_pred, r2, rmse = -all_pred, inv_r2, inv_rmse

        for element in element_order:
            mask = (an_rep == element) & diag_mask
            edge_color = "#555555" if element == 1 else "none"
            ax.scatter(
                all_ref[mask],
                final_pred[mask],
                s=1.8,
                c=element_colors[element],
                marker="o",
                alpha=0.3,
                edgecolors=edge_color,
                linewidths=0.2,
            )

        ax.plot([-1.8, 1.8], [-1.8, 1.8], "k--", alpha=0.2, lw=0.4)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.tick_params(axis="both", labelsize=4.5, pad=1, length=1.5, colors=spine_color)

        if i >= 5:
            ax.set_xlabel(r"DFT $Z^*_{\alpha\alpha}$ [e]", fontsize=7, labelpad=0.5)
        else:
            ax.set_xticklabels([])
        if i % 5 == 0:
            ax.set_ylabel(r"LES $Z^*_{\alpha\alpha}$ [e]", fontsize=7, labelpad=-1)
        else:
            ax.set_yticklabels([])

        ax.text(
            0.5,
            0.96,
            label_title + "\n" + label_level,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6,
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.7,
            f"\n $R^2$: {r2:.3f}\nRMSE: {rmse:.3f} e",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=5,
            linespacing=1.2,
        )

        inset = ax.inset_axes([0.62, 0.08, 0.35, 0.35])
        inset.set_box_aspect(1)
        for element in element_order:
            mask = (an_rep == element) & offdiag_mask
            edge_color = "#555555" if element == 1 else "none"
            inset.scatter(
                all_ref[mask],
                final_pred[mask],
                s=0.25,
                c=element_colors[element],
                marker="o",
                alpha=0.15,
                edgecolors=edge_color,
                linewidths=0.05,
            )

        inset.plot([-1, 1], [-1, 1], "k--", alpha=0.2, lw=0.3)
        inset.set_xlim(-0.8, 0.8)
        inset.set_ylim(-0.8, 0.8)
        inset.set_xticks([-1, 1])
        inset.set_yticks([-1, 1])
        inset.tick_params(axis="both", labelsize=4, pad=0.2, length=0.8, colors=spine_color)
        for spine in inset.spines.values():
            spine.set_edgecolor(spine_color)
            spine.set_linewidth(1)
        inset.text(0.05, 0.95, r"$Z^*_{\alpha\beta}$", transform=inset.transAxes, va="top", ha="left", fontsize=3.5)

    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=800, pad_inches=0.02)
    plt.close(fig)
    return output_path


def load_model_vectors(data_dir: Path, n_frames: int = 100):
    vec_energy, vec_force, vec_bec, valid_models = [], [], [], []

    for filename, display_name, theory, marker in KPCA_MODELS:
        path = data_dir / filename
        if not path.exists():
            print(f"Warning: file not found: {filename}")
            continue

        configs = read(path, index=f":{n_frames}")
        energies, forces, becs = [], [], []
        for atoms in configs:
            energy = atoms.info.get("MACE_energy")
            force = atoms.arrays.get("MACE_forces")
            bec = atoms.arrays.get("MACE_BEC")
            if energy is None or force is None or bec is None:
                continue
            energies.append(energy / len(atoms))
            forces.append(force.flatten())
            becs.append(bec.flatten())

        if not energies:
            print(f"Warning: no valid MACE_energy/MACE_forces/MACE_BEC data in {filename}")
            continue

        energy_arr = np.array(energies)
        vec_energy.append(energy_arr - np.mean(energy_arr))
        vec_force.append(np.concatenate(forces))
        vec_bec.append(np.concatenate(becs))
        valid_models.append(
            ModelStyle(
                name=display_name,
                color=THEORY_COLORS[theory],
                marker=marker,
                theory=theory,
            )
        )
        print(f"Loaded: {display_name} ({len(energies)} frames)")

    return vec_energy, vec_force, vec_bec, valid_models


def find_best_kpca_embedding(data_list: list[np.ndarray], label: str):
    x = np.stack(data_list)
    x_centered = x - np.mean(x, axis=1, keepdims=True)

    d2_matrix = pairwise_distances(x_centered, metric="euclidean", squared=True)
    mean_d2 = np.mean(d2_matrix)
    if mean_d2 < 1e-12:
        mean_d2 = 1.0

    base_gamma = 1.0 / mean_d2
    multipliers = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    n_samples = len(x_centered)
    candidates = []

    for multiplier in multipliers:
        gamma = base_gamma * multiplier
        kpca_model = SparseKPCA(
            n_components=n_samples,
            kernel={"k0": {"type": "rbf", "gamma": gamma}},
        )
        coords = kpca_model.fit_transform(x_centered)
        evals = getattr(kpca_model.kpca, "eigenvalues_", None)
        if evals is None:
            evals = getattr(kpca_model.kpca, "lambdas_", None)

        if evals is None:
            var_ratios = np.array([0.0, 0.0])
        else:
            var_ratios = evals / (np.sum(evals) + 1e-12)

        coords_2d = coords[:, [0, 1]]
        pd_dist = pairwise_distances(coords_2d, metric="euclidean")
        tri = pd_dist[np.triu_indices_from(pd_dist, k=1)]
        separation = float(np.mean(tri) + 0.75 * np.min(tri)) if tri.size > 0 else 0.0
        score = separation * (var_ratios[0] + var_ratios[1]) if label == "Energy" else separation

        candidates.append(
            {
                "coords": coords_2d,
                "var_ratios": var_ratios,
                "multiplier": multiplier,
                "score": score,
            }
        )

    best = max(candidates, key=lambda candidate: candidate["score"])
    return best["coords"], best["var_ratios"], best["multiplier"]


def plot_kpca_similarity(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_frames: int = 100,
    output_name: str = "Similarity_Map_kPCA_Final_Weighted.png",
) -> Path:
    configure_matplotlib()
    validate_inputs(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vec_energy, vec_force, vec_bec, valid_models = load_model_vectors(data_dir, n_frames=n_frames)
    if not valid_models:
        raise RuntimeError("No valid models loaded. Check data_dir and .xyz arrays.")

    fig, axes = plt.subplots(1, 3, figsize=(7, 7 / 3), dpi=500)
    fig.subplots_adjust(top=0.76, wspace=0.35)

    panel_data = [vec_energy, vec_force, vec_bec]
    panel_labels = ["Energy", "Force", "BEC"]
    kpca_results = []

    for data, panel_label in zip(panel_data, panel_labels):
        coords, var_ratios, best_mult = find_best_kpca_embedding(data, panel_label)
        kpca_results.append((coords, var_ratios))
        print(f"{panel_label}: Best RBF Gamma Multiplier={best_mult:.1f}, PC1={var_ratios[0]:.1%}")

    for ax, result, panel_label in zip(axes, kpca_results, panel_labels):
        coords, var_ratios = result
        ax.set_box_aspect(1)
        ax.grid(False)

        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        max_span = max(x_max - x_min, y_max - y_min)
        if max_span == 0:
            max_span = 1.0
        expand = 1.2
        ax.set_xlim(center_x - max_span / 2 * expand, center_x + max_span / 2 * expand)
        ax.set_ylim(center_y - max_span / 2 * expand, center_y + max_span / 2 * expand)

        jitter_amount = 0.02 * max_span
        rng = np.random.RandomState(42)

        for model_index, model in enumerate(valid_models):
            dx = rng.uniform(-jitter_amount, jitter_amount)
            dy = rng.uniform(-jitter_amount, jitter_amount)
            is_hollow = model.name == "MACELES-OFF"
            face_color = "none" if is_hollow else model.color
            edge_color = model.color if is_hollow else "white"
            line_width = 1.0 if is_hollow else 0.1
            ax.scatter(
                coords[model_index, 0] + dx,
                coords[model_index, 1] + dy,
                c=face_color,
                marker=model.marker,
                s=MARKER_SIZES[model.marker],
                edgecolors=edge_color,
                linewidths=line_width,
                zorder=3,
            )

        ax.set_xlabel(f"PC 1 ({var_ratios[0]:.1%})", labelpad=5, fontsize=10)
        ax.set_ylabel(f"PC 2 ({var_ratios[1]:.1%})", labelpad=5, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        label_x = {"Energy": 0.75, "Force": 0.8, "BEC": 0.85}.get(panel_label, 0.8)
        ax.text(label_x, 0.98, panel_label, transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold")

    category_map = {
        "RPBE-D3": "GGA",
        "RPBE": "GGA",
        "PBE": "GGA",
        "wB97M-D3": "Hybrid",
        "wB97M-V": "Hybrid",
    }
    gga_models, hybrid_models, seen = [], [], set()
    for model in valid_models:
        if model.name in seen:
            continue
        seen.add(model.name)
        if category_map.get(model.theory, "Other") == "GGA":
            gga_models.append(model)
        else:
            hybrid_models.append(model)

    def create_handle(model: ModelStyle):
        is_hollow = model.name == "MACELES-OFF"
        marker_face_color = "none" if is_hollow else model.color
        marker_edge_color = model.color if is_hollow else "white"
        marker_edge_width = 1.0 if is_hollow else 0.0
        return plt.Line2D(
            [0],
            [0],
            marker=model.marker,
            color="w",
            markerfacecolor=marker_face_color,
            markeredgecolor=marker_edge_color,
            markeredgewidth=marker_edge_width,
            markersize=np.sqrt(MARKER_SIZES[model.marker]),
            linestyle="None",
            label=model.name,
        )

    leg_gga = fig.legend(
        handles=[create_handle(model) for model in gga_models],
        title=r"$\bf{GGA\ functionals}$",
        loc="lower center",
        bbox_to_anchor=(0.20, 0.7581),
        ncol=1,
        fontsize=8,
        title_fontsize=10,
        columnspacing=1.0,
        handletextpad=0.2,
        frameon=False,
    )
    fig.legend(
        handles=[create_handle(model) for model in hybrid_models],
        title=r"$\bf{Hybrid\ functionals}$",
        loc="lower center",
        bbox_to_anchor=(0.62, 0.7581),
        ncol=3,
        fontsize=8,
        title_fontsize=10,
        columnspacing=1.0,
        handletextpad=0.2,
        frameon=False,
    )
    fig.add_artist(leg_gga)

    output_path = output_dir / output_name
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing packaged .xyz benchmark data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated figures.")
    parser.add_argument("--n-frames", type=int, default=100, help="Number of frames to read for the kPCA benchmark.")
    parser.add_argument("--only", choices=["all", "parity", "kpca"], default="all", help="Which benchmark figure(s) to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.only in {"all", "parity"}:
        parity_path = plot_bec_parity(data_dir=data_dir, output_dir=output_dir)
        print(f"Saved parity figure: {parity_path}")

    if args.only in {"all", "kpca"}:
        kpca_path = plot_kpca_similarity(data_dir=data_dir, output_dir=output_dir, n_frames=args.n_frames)
        print(f"Saved kPCA figure: {kpca_path}")


if __name__ == "__main__":
    main()
