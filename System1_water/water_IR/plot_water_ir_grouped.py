#!/usr/bin/env python3
"""Plot grouped water IR spectra from the packaged data folder.

This script is self-contained for GitHub use: all input files are resolved
relative to the script location by default.
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "water_ir_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402


MODEL_CONFIGS = [
    ("run_esen-cons-MD-500", "eSEN-OC25-sm (RPBE-D3)", "#004586FF", "RPBE-D3"),
    ("run_aqcat25-153M-500", "AQCat25-ev2 (RPBE)", "#FFD320FF", "RPBE-AQ"),
    ("run_MACE-MP0-500", "MACE-MP-0(L) (PBE)", "#71BFB2", "PBE"),
    ("run_MACE-OFF-S", "MACE-OFF23(S)", "#83CAFFFF", "wB97M-D3"),
    ("run_MACE-OFF-M", "MACE-OFF23(M)", "#0084D1FF", "wB97M-D3"),
    ("run_UMA-M-MD-500", "UMA-M(OMol)", "#d9042b", "wB97M-V"),
    ("run_MACE-omol-MD-500", "MACE-OMol", "#f27b50", "wB97M-V"),
    ("run_MACE-MH-MD-500", "MACE-MH-1(OMol)", "#4B1F6FFF", "wB97M-V"),
    ("run_orb-omol-500", "Orb-v3-OrbMol", "#fb6095", "wB97M-V"),
    ("run_orb-omol-direct-500", "Orb-v3-OrbMol-d", "#fccde5", "wB97M-V"),
]

GROUP_DEFS = [
    ("GGA", {"PBE", "RPBE", "RPBE-D3", "RPBE-AQ"}),
    (r"Hybrid $\omega$B97M-D3", {"wB97M-D3"}),
    (r"Hybrid $\omega$B97M-V", {"wB97M-V"}),
]

THEORY_LINESTYLES = {
    "RPBE-D3": (0, (1, 1)),
    "RPBE-AQ": (0, (3, 1, 1, 1, 1, 1)),
    "RPBE": (0, (4, 1.5)),
    "PBE": (0, (3, 1, 1, 1)),
    "wB97M-V": "solid",
    "wB97M-D3": "--",
}

MODEL_LINESTYLES = {
    "run_esen-cons-MD-500": (0, (1, 1)),
    "run_aqcat25-153M-500": (0, (3, 1, 1, 1, 1, 1)),
    "run_MACE-MP0-500": (0, (4, 1.5)),
    "run_MACE-OFF-S": "-",
    "run_MACE-OFF-M": (0, (5, 1.5)),
    "run_UMA-M-MD-500": "-",
    "run_MACE-omol-MD-500": "--",
    "run_MACE-MH-MD-500": "-.",
    "run_orb-omol-500": ":",
    "run_orb-omol-direct-500": (0, (3, 1, 1, 1)),
}


def apply_nqe_correction(omega: np.ndarray) -> np.ndarray:
    omega_real = np.real(omega)
    w_anchors = [0, 1000, 1640, 2800, 3450, 4000]
    s_anchors = [5, 5, 60, 100, 175, 175]
    shifts = np.interp(omega_real, w_anchors, s_anchors)
    return omega_real - shifts


def normalize_intensity(y: np.ndarray) -> np.ndarray:
    y_real = np.real(np.asarray(y))
    y_base = y_real - np.min(y_real)
    y_max = np.max(y_base)
    if y_max <= 1e-12:
        return np.zeros_like(y_base)
    return y_base / y_max


def fftcrosscorr(x: np.ndarray, y: np.ndarray, dlen: int = 10000) -> np.ndarray:
    if dlen % 2 == 0:
        dlen -= 1

    dt = x[1, 0] - x[0, 0]
    window = len(x) // dlen
    omega0 = 2.0 * np.pi / (dlen - 1) / dt

    cxyomega = np.zeros((dlen, 2), dtype=np.complex128)
    cxyomega[0 : dlen // 2 + 1, 0] = np.arange(dlen // 2 + 1) * omega0
    cxyomega[dlen // 2 + 1 :, 0] = np.arange(dlen // 2, 0, -1) * omega0 * -1

    for i in range(window):
        dx = x[i * dlen : (i + 1) * dlen, 1]
        dy = y[i * dlen : (i + 1) * dlen, 1]
        ax_fft = np.fft.fft(dx, axis=0)
        ay_fft = np.fft.fft(dy, axis=0)
        cxyomega[:, 1] += np.conjugate(ax_fft) * ay_fft / dlen * dt

    for i in range(window - 1):
        dx = x[i * dlen + dlen // 2 : (i + 1) * dlen + dlen // 2, 1]
        dy = y[i * dlen + dlen // 2 : (i + 1) * dlen + dlen // 2, 1]
        ax_fft = np.fft.fft(dx, axis=0)
        ay_fft = np.fft.fft(dy, axis=0)
        cxyomega[:, 1] += np.conjugate(ax_fft) * ay_fft / dlen * dt

    cxyomega[:, 1] /= window * 2 - 1
    return cxyomega


def smooth_signal(signal: np.ndarray, window_size: int = 51, sigma: float = 7.0) -> np.ndarray:
    kernel = gaussian(window_size, std=sigma)
    kernel /= np.sum(kernel)
    return convolve(signal, kernel, mode="same")


def compute_ir_spectrum(
    total_dp: np.ndarray,
    dt: float = 0.50,
    dlen: int = 10000,
    length: int = 800,
    window_size: int = 30,
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dpt = np.zeros((len(total_dp), 4))
    dpt[:, 0] = np.arange(len(total_dp)) * dt
    dpt[:, 1:4] = total_dp

    ft_x = fftcrosscorr(dpt[:, [0, 1]], dpt[:, [0, 1]], dlen=dlen)
    ft_y = fftcrosscorr(dpt[:, [0, 2]], dpt[:, [0, 2]], dlen=dlen)
    ft_z = fftcrosscorr(dpt[:, [0, 3]], dpt[:, [0, 3]], dlen=dlen)

    omega = ft_x[:, 0] * 1e15 / 2.99792458e10 / (2 * np.pi)
    ft_avg = (ft_x[:, 1] + ft_y[:, 1] + ft_z[:, 1]) / 3
    smooth_inten = smooth_signal(ft_avg, window_size=window_size, sigma=sigma)
    return omega[:length], ft_avg[:length], smooth_inten[:length]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def model_pickle_path(data_dir: Path, folder: str) -> Path:
    return data_dir / "model_spectra" / folder / "bec_dict.pkl"


def required_input_paths(data_dir: Path) -> list[Path]:
    paths = [model_pickle_path(data_dir, folder) for folder, *_ in MODEL_CONFIGS]
    paths.extend(
        [
            model_pickle_path(data_dir, "run_MACE-OFF24M"),
            data_dir / "experimental" / "water_IR.pkl",
            data_dir / "reference" / "MACE-OFF23S_MD_ref.csv",
            data_dir / "reference" / "MACE-OFF23S_PIMD.csv",
        ]
    )
    return paths


def validate_inputs(data_dir: Path, allow_missing: bool = False) -> None:
    missing = [path for path in required_input_paths(data_dir) if not path.exists()]
    if missing and not allow_missing:
        rel_paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{rel_paths}")


def configure_matplotlib() -> None:
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
        }
    )


def load_model_spectra(data_dir: Path) -> list[dict[str, object]]:
    spectra = []
    for folder, label, color, theory in MODEL_CONFIGS:
        path = model_pickle_path(data_dir, folder)
        if not path.exists():
            continue

        bec_dict = load_pickle(path)
        omega, _, intensity = compute_ir_spectrum(
            bec_dict["total_dp"],
            dt=0.25,
            dlen=5000,
            length=400,
            window_size=20,
            sigma=1.5,
        )

        spectra.append(
            {
                "folder": folder,
                "theory": theory,
                "label": label,
                "color": color,
                "omega": apply_nqe_correction(np.real(omega)),
                "intensity": normalize_intensity(intensity),
                "linestyle": MODEL_LINESTYLES.get(folder, THEORY_LINESTYLES.get(theory, "-")),
            }
        )
    return spectra


def load_csv_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    omega = np.real(df.iloc[:, 0].values)
    intensity = normalize_intensity(df.iloc[:, 1].values)
    return omega, intensity


def plot_off24m_reference(ax: plt.Axes, data_dir: Path) -> None:
    path = model_pickle_path(data_dir, "run_MACE-OFF24M")
    if not path.exists():
        return

    bec_dict = load_pickle(path)
    omega, _, intensity = compute_ir_spectrum(
        bec_dict["total_dp"],
        dt=0.25,
        dlen=5000,
        length=400,
        window_size=20,
        sigma=1.5,
    )
    ax.plot(
        apply_nqe_correction(np.real(omega)),
        normalize_intensity(intensity),
        color="#03045e",
        linestyle=":",
        linewidth=2.0,
        label="MACE-OFF24(M)",
        zorder=5,
    )


def plot_kovacs_references(ax: plt.Axes, data_dir: Path) -> None:
    md_path = data_dir / "reference" / "MACE-OFF23S_MD_ref.csv"
    if md_path.exists():
        md_omega, md_intensity = load_csv_spectrum(md_path)
        ax.plot(
            apply_nqe_correction(md_omega),
            md_intensity,
            color="#b5ea8c",
            linestyle="-.",
            linewidth=1.5,
            label="MACE-OFF23(S) MD (Kovacs et al., 2025)",
            zorder=1,
        )

    pimd_path = data_dir / "reference" / "MACE-OFF23S_PIMD.csv"
    if pimd_path.exists():
        pimd_omega, pimd_intensity = load_csv_spectrum(pimd_path)
        ax.plot(
            pimd_omega,
            pimd_intensity,
            color="#646f77",
            linestyle="--",
            linewidth=1.5,
            label="MACE-OFF23(S) PIGS (Kovacs et al., 2025)",
            zorder=1,
        )


def format_axis(ax: plt.Axes, group_name: str) -> None:
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))
    ax.set_ylabel("Intensity [a.u.]", labelpad=3, fontsize=11)
    ax.text(
        0.97,
        0.98,
        group_name,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )


def draw_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return

    paired = list(zip(handles, labels))
    non_exp = [pair for pair in paired if pair[1] != "Exp"]
    exp_only = [pair for pair in paired if pair[1] == "Exp"]
    ordered = non_exp + exp_only

    ax.legend(
        [pair[0] for pair in ordered],
        [pair[1] for pair in ordered],
        frameon=False,
        loc="upper left",
        fontsize=9,
        ncol=1,
        columnspacing=0.6,
        handletextpad=0.4,
        handlelength=2.2,
        labelspacing=0.3,
    )


def plot_grouped_ir(data_dir: Path, output: Path, dpi: int = 500) -> None:
    configure_matplotlib()
    validate_inputs(data_dir)

    exp_data = load_pickle(data_dir / "experimental" / "water_IR.pkl")
    all_spectra = load_model_spectra(data_dir)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.27 * 0.75, 9.2),
        dpi=300,
        sharex=True,
    )

    for idx, (ax, (group_name, group_theories)) in enumerate(zip(axes, GROUP_DEFS)):
        if "Exp" in exp_data:
            exp_omega = exp_data["Exp"][0]
            exp_intensity = normalize_intensity(exp_data["Exp"][1])
            ax.fill_between(
                exp_omega,
                exp_intensity,
                color="#666666",
                alpha=0.2,
                edgecolor="none",
                label="Exp" if idx == 0 else "_nolegend_",
                zorder=1,
            )

        for spectrum in all_spectra:
            if spectrum["theory"] not in group_theories:
                continue
            ax.plot(
                spectrum["omega"],
                spectrum["intensity"],
                color=spectrum["color"],
                linestyle=spectrum["linestyle"],
                linewidth=2.0,
                label=spectrum["label"],
                zorder=5,
            )

        if "wB97M-D3" in group_theories:
            plot_off24m_reference(ax, data_dir)
            plot_kovacs_references(ax, data_dir)

        format_axis(ax, group_name)
        draw_legend(ax)

    axes[-1].set_xlabel(r"Wavenumber [cm$^{-1}$]", labelpad=3, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "Water_IR_data",
        help="Directory containing model_spectra/, experimental/, and reference/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "IR_plot_water_grouped.png",
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=500, help="Output image DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_grouped_ir(args.data_dir.resolve(), args.output.resolve(), dpi=args.dpi)


if __name__ == "__main__":
    main()
