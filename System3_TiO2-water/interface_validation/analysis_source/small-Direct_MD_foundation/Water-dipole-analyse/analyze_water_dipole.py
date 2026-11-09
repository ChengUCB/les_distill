#!/usr/bin/env python3
"""Water dipole orientation vs distance from the TiO2 surface, all trajectories.

Dipole convention (charges O: -2, H: +1): mu = sum q_i r_i = 2*(H_center - O),
i.e. the dipole points from O toward the H midpoint — identical direction to
the O->H-center vector used in the legacy get_dipole_his scripts.

cos(theta) = mu_hat . n_hat, where n_hat is the INWARD surface normal (from the
nearest TiO2 surface into the water): +z for the bottom interface, -z for the
top. cos>0 means H atoms point away from the surface (into the film); cos<0
means H atoms point toward the surface.

Processing matches the density pipeline: final 75% of frames, per-frame planar
surface planes (legacy tol, same tol values as the density analysis), distance
of the water O to its nearest surface, both interfaces accumulated, 5-block SEM.

Water molecules are (re)constructed every frame: each H is assigned to its
nearest O (periodic, cKDTree); a water molecule is a water-classified O with
exactly two H. O with one/zero H (hydroxyl/oxide) are counted but excluded from
the dipole statistics; the dissociation fraction is reported.

Outputs per trajectory: <cos theta>(dz) profile + molecule count density
(.dat), P(cos theta) distributions in layer windows, and comparison figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "Water-dipole-analyse"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "water_density_analysis"))
from trajectory_io import (  # noqa: E402
    open_text,
    resolve_compressed,
    source_identity,
    uncompressed_size,
)
import analyze_dft_system_size_normalization as core  # noqa: E402

OUT = HOME / "results"
OUT.mkdir(exist_ok=True)
CACHE = HOME / "pos_cache"
CACHE.mkdir(exist_ok=True)
CACHE_VERSION = 2

# trajectory -> (path, surface tol from the density analysis)
# NOTE (2026-07-16): the raw xyz for the small-cell trajectories and the DFT
# references were removed from this machine; those entries are served from
# pos_cache/.  When raw data exists, load_full validates source provenance;
# otherwise it deliberately falls back to the irreplaceable cache.
TRAJS = {
    "GemNet_student_small": (ROOT / "Small-MD-CACELES/GemNet-OC22/md_out_CACELES-200ps.xyz", 0.75),
    "GemNet_foundation_small": (ROOT / "Small-directMD-foundation/GemNet-OC22/md_out-2025-12.xyz", 0.74),
    "GemNet_large": (ROOT / "Large-student-MD/GemNet-OC22/md_out.xyz", 0.80),
    "MH_student_small": (ROOT / "Small-MD-CACELES/MACE-MH-omat/md_out_CACELES-200ps.xyz", 0.75),
    "MH_foundation_small": (ROOT / "Small-directMD-foundation/MACE-MH-omat/md_out-2025-12.xyz", 0.73),
    "GemNet_foundation_mid": (ROOT / "Mid-MD_foundation/GemNet-OC22/md.xyz", 0.74),
    "MH_foundation_mid": (ROOT / "Mid-MD_foundation/MACE-MH-omat/md.xyz", 0.74),
    "MP_foundation_mid": (ROOT / "Mid-MD_foundation/MACE-MP0a_L/md.xyz", 0.74),
    "GemNet_mid": (ROOT / "Mid-MD-CACELES/GemNet-OC22-larger/md_out_CACELES-200ps.xyz", 0.74),
    "SCAN_student_mid": (ROOT / "Mid-MD-CACELES/CACELES-SCAN-larger/md_out_CACELES-200ps.xyz", 0.74),
    "SCAN_student_large": (ROOT / "Large-student-MD/CACELES-SCAN-DFT/md_out.xyz", 0.80),
    "CACE_T0_large": (ROOT / "Large-student-MD/CACE-SCANT0/md_out.xyz", 0.80),
    "CACE_T0_mid": (ROOT / "Mid-MD-CACELES/CACE-T0-larger/md_out_CACELES-200ps.xyz", 0.74),
    "MP_finetuned_mid": (ROOT / "Mid-MD-CACELES/MACE-MP-finetuned-larger/md_out_CACELES-200ps.xyz", 0.74),
    "MP_finetuned_10_large": (ROOT / "Large-student-MD/MACE-MP-finetuning-10/md_out_finetuning_10.xyz", 0.80),
    "MP_finetuned_50_large": (ROOT / "Large-student-MD/MACE-MP-finetuning-50/md_out_finetuning_50.xyz", 0.80),
    "MH_mid": (ROOT / "Mid-MD-CACELES/MACE-MH-omat_larger/md_out_CACELES-200ps.xyz", 0.74),
    "MH_large": (ROOT / "Large-student-MD/MACE-MH-omat/md_out-2025-12.xyz", 0.80),
    "MP_student_small": (ROOT / "Small-MD-CACELES/MACE-MP0a_L/md_out_CACELES-200ps.xyz", 0.75),
    "MP_foundation_small": (ROOT / "Small-directMD-foundation/MACE-MP0a_L/md_out_continue-200ps.xyz", 0.72),
    "MP_mid": (ROOT / "Mid-MD-CACELES/MACE-MP0a_L_larger/md_out_CACELES-200ps.xyz", 0.74),
    "MP_large": (ROOT / "Large-student-MD/MACE-MP0a_L/md_out-2025-12.xyz", 0.80),
    "DFT_small": (ROOT / "Small-MD-CACELES/DFT-ref/md_out-2025-12.xyz", 0.77),
    "DFT_large": (ROOT / "Large-student-MD/DFT-ref/md_out.xyz", 0.80),
}

DZ_BIN = 0.10
DZ_MAX = 12.0
LAYERS = {"layer1": (0.8, 2.0), "layer2": (2.0, 3.0), "center": (5.0, 10.0)}
COS_BINS = np.linspace(-1, 1, 41)
NBLOCKS = 5


def _write_cache(
    npz: Path,
    path: Path,
    pos: np.ndarray,
    symbols: np.ndarray,
    box: np.ndarray,
    natoms: int,
    nframes: int,
) -> None:
    """Atomically write positions and source-provenance metadata."""
    source_path, source_size, source_mtime_ns = source_identity(path)
    tmp = npz.with_name(f".{npz.name}.tmp.npz")
    np.savez_compressed(
        tmp,
        pos=pos,
        symbols=symbols,
        box=box,
        cache_version=np.asarray(CACHE_VERSION, dtype=np.int64),
        source_path=np.asarray(source_path),
        source_size=np.asarray(source_size, dtype=np.int64),
        source_mtime_ns=np.asarray(source_mtime_ns, dtype=np.int64),
        source_natoms=np.asarray(natoms, dtype=np.int64),
        source_nframes=np.asarray(nframes, dtype=np.int64),
        keep_fraction=np.asarray(core.KEEP_FRACTION, dtype=float),
    )
    tmp.replace(npz)


def _load_cache(npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a cache eagerly so the returned arrays do not retain an open file."""
    with np.load(npz, allow_pickle=False) as data:
        pos = data["pos"]
        symbols = data["symbols"]
        box = data["box"]
        metadata = {
            key: data[key].item()
            for key in (
                "cache_version", "source_path", "source_size", "source_mtime_ns",
                "source_natoms", "source_nframes", "keep_fraction",
            )
            if key in data.files
        }
    return pos, symbols, box, metadata


def load_full(name: str):
    """Final-75% positions with automatic invalidation of stale caches.

    A raw trajectory is authoritative whenever it is available.  Cache-only
    loading remains supported for the removed small-cell and DFT trajectories.
    """
    npz = CACHE / f"{name}.npz"
    path, _ = TRAJS[name]
    path = resolve_compressed(path)

    if npz.exists():
        pos, symbols, box, metadata = _load_cache(npz)
        if not path.exists():
            # Several irreplaceable small/DFT raw trajectories were removed.
            return pos, symbols, box

        stat = path.stat()
        metadata_current = (
            metadata.get("cache_version") == CACHE_VERSION
            and metadata.get("source_size") == uncompressed_size(path)
            and metadata.get("source_mtime_ns") == stat.st_mtime_ns
            and metadata.get("keep_fraction") == core.KEEP_FRACTION
        )
        if metadata_current:
            expected_keep = int(
                np.floor(metadata["source_nframes"] * core.KEEP_FRACTION)
            )
            if pos.shape == (expected_keep, metadata["source_natoms"], 3):
                return pos, symbols, box

        # Legacy caches can be upgraded without rebuilding if their timestamp
        # and array shape agree with the current raw trajectory.
        if not metadata:
            natoms, nframes = core.trajectory_shape(path)
            expected_keep = int(np.floor(nframes * core.KEEP_FRACTION))
            legacy_current = (
                npz.stat().st_mtime_ns >= stat.st_mtime_ns
                and pos.shape == (expected_keep, natoms, 3)
                and symbols.shape == (natoms,)
            )
            if legacy_current:
                _write_cache(npz, path, pos, symbols, box, natoms, nframes)
                return pos, symbols, box

        print(f"cache stale for {name}; rebuilding from {path}", flush=True)

    natoms, nframes = core.trajectory_shape(path)
    keep = int(np.floor(nframes * core.KEEP_FRACTION))
    start = nframes - keep
    pos = np.empty((keep, natoms, 3), dtype=np.float32)
    symbols = np.empty(natoms, dtype="U2")
    box = None
    with open_text(path) as fh:
        for fi in range(nframes):
            fh.readline()
            comment = fh.readline()
            if fi < start:
                for _ in range(natoms):
                    fh.readline()
                continue
            k = fi - start
            if k == 0:
                cell = core.parse_lattice(comment)
                box = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
            for ai in range(natoms):
                f = fh.readline().split()
                if k == 0:
                    symbols[ai] = f[0]
                pos[k, ai] = [float(v) for v in f[1:4]]
    _write_cache(npz, path, pos, symbols, box, natoms, nframes)
    return pos, symbols, box


def analyze(name: str):
    pos, symbols, box = load_full(name)
    tol = TRAJS[name][1]
    o_idx = np.flatnonzero(symbols == "O")
    h_idx = np.flatnonzero(symbols == "H")
    ti_idx = np.flatnonzero(symbols == "Ti")

    # classify water O / slab from first selected frame (same rule as density)
    slab_idx, water_o_idx = core.classify_atoms(
        symbols, pos[0].astype(float),
        np.diag(box),
    )
    water_set = np.zeros(len(symbols), dtype=bool)
    water_set[water_o_idx] = True
    is_water_o = water_set[o_idx]  # mask over the O list

    edges = np.arange(0.0, DZ_MAX + 0.5 * DZ_BIN, DZ_BIN)
    zc = 0.5 * (edges[:-1] + edges[1:])
    nbin = len(zc)

    frame_cos_sum = []
    frame_cos_cnt = []
    layer_hists = {k: np.zeros(len(COS_BINS) - 1) for k in LAYERS}
    layer_counts = {k: 0 for k in LAYERS}
    dissoc = []

    for k in range(len(pos)):
        p = pos[k].astype(float)
        # surface planes (same as density pipeline)
        slab_z = p[slab_idx, 2]
        zm = slab_z.mean()
        lower = slab_z[slab_z < zm]
        upper = slab_z[slab_z >= zm]
        bottom = lower[lower > lower.max() - tol].mean()
        top = upper[upper < upper.min() + tol].mean()
        mid = 0.5 * (bottom + top)

        # H -> nearest O (periodic)
        o_pos = np.mod(p[o_idx], box)
        h_pos = np.mod(p[h_idx], box)
        tree = cKDTree(o_pos, boxsize=box)
        _, nearest = tree.query(h_pos)
        counts = np.bincount(nearest, minlength=len(o_idx))

        intact = np.flatnonzero(is_water_o & (counts == 2))
        dissoc.append(1.0 - intact.size / max(is_water_o.sum(), 1))

        # gather the two H of each intact water O
        order = np.argsort(nearest, kind="stable")
        sorted_owner = nearest[order]
        first = np.searchsorted(sorted_owner, intact, side="left")
        h1 = h_idx[order[first]]
        h2 = h_idx[order[first + 1]]
        o_at = o_idx[intact]

        o_r = p[o_at]
        d1 = p[h1] - o_r
        d2 = p[h2] - o_r
        d1 -= box * np.round(d1 / box)
        d2 -= box * np.round(d2 / box)
        mu = 0.5 * (d1 + d2)  # direction of 2*(Hc - O)
        mu_norm = np.linalg.norm(mu, axis=1)
        ok = mu_norm > 1e-8
        o_z = o_r[:, 2]
        on_bottom = o_z <= mid
        cos = np.where(on_bottom, mu[:, 2], -mu[:, 2]) / np.maximum(mu_norm, 1e-12)
        dz = np.where(on_bottom, o_z - bottom, top - o_z)
        cos, dz = cos[ok], dz[ok]

        ib = np.clip(((dz - 0.0) / DZ_BIN).astype(int), 0, nbin - 1)
        valid = (dz >= 0) & (dz < DZ_MAX)
        s = np.bincount(ib[valid], weights=cos[valid], minlength=nbin)
        c = np.bincount(ib[valid], minlength=nbin).astype(float)
        frame_cos_sum.append(s)
        frame_cos_cnt.append(c)

        for lname, (lo, hi) in LAYERS.items():
            m = (dz >= lo) & (dz < hi)
            if m.any():
                layer_hists[lname] += np.histogram(cos[m], bins=COS_BINS)[0]
                layer_counts[lname] += int(m.sum())

    S = np.asarray(frame_cos_sum)
    C = np.asarray(frame_cos_cnt)
    mean_cos = np.divide(S.sum(0), C.sum(0), out=np.zeros(nbin), where=C.sum(0) > 0)

    # block SEM over per-block mean-cos profiles
    block_means = []
    for bs, bc in zip(np.array_split(S, NBLOCKS), np.array_split(C, NBLOCKS)):
        bm = np.divide(bs.sum(0), bc.sum(0), out=np.zeros(nbin), where=bc.sum(0) > 0)
        block_means.append(bm)
    block_means = np.asarray(block_means)
    sem = block_means.std(axis=0, ddof=1) / np.sqrt(NBLOCKS)

    count_density = C.sum(0) / len(pos) / DZ_BIN  # molecules per A per frame (2 interfaces)
    low_stat = C.sum(0) < 50
    mean_cos_masked = mean_cos.copy()
    mean_cos_masked[low_stat] = np.nan

    smooth = savgol_filter(np.nan_to_num(mean_cos), 5, 2)
    smooth[low_stat] = np.nan

    dist = {k: layer_hists[k] / max(layer_counts[k], 1) / (COS_BINS[1] - COS_BINS[0])
            for k in LAYERS}

    np.savetxt(
        OUT / f"{name}_dipole_profile.dat",
        np.column_stack((zc, mean_cos_masked, sem, count_density)),
        fmt="%.6f",
        header="dz_A mean_cos_theta block_SEM molecule_count_per_A_per_frame "
               "(cos>0: H away from surface; O=-2 H=+1 dipole)",
    )
    return {
        "z": zc, "mean": mean_cos_masked, "smooth": smooth, "sem": sem,
        "count": count_density, "dist": dist, "layer_counts": dict(layer_counts),
        "dissociation_fraction": float(np.mean(dissoc)),
        "n_waterO": int(is_water_o.sum()), "frames": len(pos),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for name in TRAJS:
        print(f"analyzing {name} ...", flush=True)
        results[name] = analyze(name)
        r = results[name]
        print(f"  waterO={r['n_waterO']} frames={r['frames']} "
              f"dissociated={r['dissociation_fraction']:.1%}", flush=True)

    # ---------- figure 1: <cos theta>(dz) size comparison ----------
    SERIES = {
        "MACE-MH-Omat": [("MH_student_small", "small (20 Å)", ":", None),
                          ("MH_mid", "mid (30 Å)", "-.", None),
                          ("MH_large", "large (67 Å)", "--", "0.15")],
        "MACE-MP0a_L": [("MP_student_small", "small (20 Å)", ":", None),
                         ("MP_mid", "mid (30 Å)", "-.", None),
                         ("MP_large", "large (67 Å)", "--", "0.15")],
        "GemNet-OC22": [("GemNet_student_small", "small (20 Å)", ":", None),
                         ("GemNet_large", "large (67 Å)", "--", "0.15")],
        "DFT reference": [("DFT_small", "small (20 Å)", ":", None),
                           ("DFT_large", "large (67 Å)", "--", "0.15")],
    }
    BASECOL = {"MACE-MH-Omat": "#d95f02", "MACE-MP0a_L": "#1b9e77",
               "GemNet-OC22": "#2878b5", "DFT reference": "#7b3294"}

    fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.2), dpi=180, sharex=True)
    for ax, (title, entries) in zip(axes, SERIES.items()):
        color = BASECOL[title]
        for name, label, ls, over in entries:
            r = results[name]
            c = over or color
            lw = 2.4 if "mid" in name else 1.9
            ax.fill_between(r["z"], r["mean"] - r["sem"], r["mean"] + r["sem"],
                            color=c, alpha=0.12, lw=0)
            ax.plot(r["z"], r["mean"], color=c, ls=ls, lw=lw, label=label)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axvspan(0.8, 2.0, color="#2166ac", alpha=0.05)
        ax.axvspan(2.0, 3.0, color="#b2182b", alpha=0.05)
        ax.set(xlim=(0, 10), ylim=(-1.0, 1.0),
               ylabel=r"$\langle\cos\theta\rangle$", title=title)
        ax.legend(frameon=False, fontsize=8.3, loc="upper right")
        ax.grid(ls="--", alpha=0.2)
    axes[-1].set_xlabel(r"Distance from TiO$_2$ surface, $\Delta z$ [Å]")
    fig.suptitle(
        "Water dipole orientation profiles (final 75% frames; O:-2 H:+1;\n"
        r"$\cos\theta>0$: H point away from the surface; shading $\pm$1 block SEM)",
        y=0.998, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "dipole_orientation_profiles.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 2: P(cos theta) by layer ----------
    cosc = 0.5 * (COS_BINS[:-1] + COS_BINS[1:])
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 13.0), dpi=180, sharex=True)
    for row, (title, entries) in enumerate(SERIES.items()):
        color = BASECOL[title]
        for coli, lname in enumerate(LAYERS):
            ax = axes[row, coli]
            for name, label, ls, over in entries:
                r = results[name]
                c = over or color
                ax.plot(cosc, r["dist"][lname], color=c, ls=ls,
                        lw=2.2 if "mid" in name else 1.7, label=label)
            ax.axvline(0, color="0.75", lw=0.7)
            lo, hi = LAYERS[lname]
            if row == 0:
                ax.set_title(f"{lname}  ({lo:.1f}–{hi:.1f} Å)")
            if coli == 0:
                ax.set_ylabel(f"{title}\n" + r"$P(\cos\theta)$", fontsize=9)
            if row == 3:
                ax.set_xlabel(r"$\cos\theta$")
            ax.grid(ls="--", alpha=0.2)
            if row == 0 and coli == 2:
                ax.legend(frameon=False, fontsize=7.8)
    fig.suptitle("Water dipole orientation distributions by layer "
                 "(O:-2 H:+1; cos>0: H away from surface)", y=0.999, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "dipole_orientation_distributions.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- summary ----------
    lines = ["# Water dipole orientation summary",
             "",
             "cos>0: H away from surface (dipole pointing into the film).",
             "",
             "| Trajectory | waterO | frames | dissociated | <cos> layer1 | <cos> layer2 | <cos> center |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in TRAJS:
        r = results[name]
        means = []
        for lname, (lo, hi) in LAYERS.items():
            m = (r["z"] >= lo) & (r["z"] < hi) & np.isfinite(r["mean"]) & (r["count"] > 0)
            w = r["count"][m]
            means.append(float(np.average(r["mean"][m], weights=w)) if m.any() else np.nan)
        lines.append(f"| {name} | {r['n_waterO']} | {r['frames']} | "
                     f"{r['dissociation_fraction']:.1%} | "
                     f"{means[0]:+.3f} | {means[1]:+.3f} | {means[2]:+.3f} |")
    (OUT / "DIPOLE_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nResults in {OUT}", flush=True)


if __name__ == "__main__":
    main()
