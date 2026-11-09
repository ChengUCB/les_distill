# Teacher-label comparison: Supplemental Figures S9 and S10

Data and plotting scripts for the energy-label comparison (S9) and Cartesian-force-label comparison (S10) in *Distilling latent electrostatics from foundation machine learning interatomic potentials*, DOI [10.1103/kgym-z2rw](https://doi.org/10.1103/kgym-z2rw).

Both figures compare GemNet-OC22 and MACE-MP-0(L) with MACE-MH-1(OMat) on the same 3,168 anatase TiO2(101)-water configurations, with 426 atoms per configuration. MACE-MH-1(OMat) is the common comparison baseline, not a ground-truth reference. This package reproduces the stored-label analysis; it does not rerun the teacher models or train student models.

## Contents

| Path | Purpose |
| --- | --- |
| `data/cache_SCAN.npz` | Canonical frame order, geometry, atom counts, species, cells, PBC, and SCAN labels for all 3,346 original cached configurations. |
| `data/cache_GemNet.npz` | GemNet-OC22 geometries and energy/force labels. |
| `data/cache_MACE_MH.npz` | MACE-MH-1(OMat) geometries and energy/force labels. |
| `data/cache_MACE_MP.npz` | MACE-MP-0(L) geometries and energy/force labels. |
| `data/pairing.npz` | Frame-order maps from the canonical SCAN order to each model cache. |
| `data/environment_analysis/centered_energies.csv` | Preserved audit export used to cross-check the S9 energy centering. |
| `scripts/` | Portable plotting, PDF export, and verification commands. |
| `reference/` | Unchanged final PNGs, numerical metrics, S9 CSV, and manuscript figure PDFs. |
| `provenance.json` | Original XYZ basenames and recorded checksums, source-script hashes, and a record of packaging changes. |
| `checksums.sha256` | SHA-256 checksums of the curated package files. |
| `validation.json` | Recorded numerical and image checks from package preparation. |

The four cache files, pairing file, and audit CSV were copied without changing their bytes. They are numerical analysis caches, not the original extended-XYZ files. The original XYZ basenames/checksums are recorded in `provenance.json`; those original files are not bundled here. Cache fields preserve positions, elements, energy, forces, cell, PBC, and frame boundaries, but are not a claim to preserve every possible XYZ metadata field.

`cache_SCAN.npz` is a required dependency of the preserved plotting method: it selects the 426-atom frames and defines the canonical atom ordering. Its SCAN energy/force values are **not plotted** in S9 or S10. The other 178 configurations remain in the byte-preserved caches but are excluded by the `n == 426` selection.

## Download and environment

The large `data/cache_*.npz` files must be downloaded through Git LFS, not left as small text pointer files. From the repository root after cloning:

```bash
git lfs install
git lfs pull --include="System3_TiO2-water/teacher_label_comparison/data/cache_*.npz"
```

Then, from this directory, create an environment and install the versions used for validation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_reproduction.py --skip-figures
```

The plotting dependencies are NumPy, pandas, Matplotlib, and Pillow. No trained-model pickle, Torch installation, teacher checkpoint, external drive, or network access is needed for reproduction once the files and dependencies have been downloaded. The verification command checks file integrity and the frame/atom geometry mapping before plotting.

## Reproduce the figures

```bash
python scripts/gen_fig_full_interface_energy_parity.py
python scripts/gen_fig_full_interface_force_parity.py
python scripts/export_prerendered_parity_pdfs.py
python scripts/verify_reproduction.py
```

Outputs are written to `reproduced/` by default. The scripts find `data/` relative to their own location and can be invoked from another working directory. To select a different output directory, pass the same directory to the commands:

```bash
python scripts/gen_fig_full_interface_energy_parity.py --output-dir results
python scripts/gen_fig_full_interface_force_parity.py --output-dir results
python scripts/export_prerendered_parity_pdfs.py --input-dir results
python scripts/verify_reproduction.py --output-dir results
```

The plotting scripts also accept `--data-dir` for an equivalent relocated cache directory. Verification always uses the bundled package data and reference files. Plotting S10 processes all 4,048,704 force components per panel, so it uses more memory and time than S9. No random subsampling is performed.

S9 produces a 3,168-row energy CSV, metrics JSON, caption text, PNG, and PDF. S10 produces metrics JSON, caption text, PNG, and PDF; the component-level force values remain in the caches to avoid duplicating several million rows in a CSV.

## Definitions and units

For each model `m`, the S9 plotted energy is

```text
relative_energy[m, i] = 1000 * (E[m, i] - mean(E[m, selected_frames])) / 426
```

`E` is total energy in eV. The result is in meV/atom. Each model has its own mean over exactly the selected 3,168 configurations. There is no fitted slope or multiplicative correction. The main limits are -60 to 90 meV/atom; both insets use -20 to 10 meV/atom.

S10 plots signed `Fx`, `Fy`, and `Fz` labels in eV/Angstrom on the same matched atoms. It includes all Ti, O, and H atoms: `3168 * 426 * 3 = 4,048,704` components per panel. Forces are not centered, rescaled, projected, adjusted to remove net force, or subsampled. Main limits are -10 to 10 eV/Angstrom; both insets use -1 to 1 eV/Angstrom.

RMSD is `sqrt(mean((compared_model - MACE_MH)**2))`. Force RMSDs are displayed in meV/Angstrom, with the factor of 1000 applied after the RMSD is computed from eV/Angstrom labels.

| Compared model | S9 RMSD (meV/atom) | S10 RMSD (meV/Angstrom) |
| --- | ---: | ---: |
| GemNet-OC22 | 2.9501462097072437 | 80.2049422318814 |
| MACE-MP-0(L) | 0.6851123186091987 | 81.8888488148665 |

## Cache schema and pairing

All NPZ files are read with `allow_pickle=False`.

| Field | Shape | Meaning / units |
| --- | --- | --- |
| `n` | `(3346,)` | Number of atoms in each cached frame. |
| `offsets` | `(3347,)` | Frame boundaries into the concatenated per-atom arrays. |
| `z` | `(1388826,)` | Atomic numbers: H=1, O=8, Ti=22. |
| `pos` | `(1388826, 3)` | Cartesian positions, Angstrom. |
| `force` | `(1388826, 3)` | Cartesian force labels, eV/Angstrom. |
| `cell` | `(3346, 3, 3)` | Cell vectors, Angstrom. |
| `energy` | `(3346,)` | Total energy labels, eV. |
| `pbc` | `(3346, 3)` | Periodic-boundary flags. |
| `counts` | `(3346, 3)` | Per-frame species counts from the original cache; unused in these plots. |

For canonical frame `i`, `pairing[model][i]` gives its row in that model's cache. Atom indices within each matched frame are preserved. The verification utility checks that every map is a full permutation, selects the 3,168 frames, and compares their atom counts, species order, Cartesian coordinates, cell matrices, and PBC flags. Coordinate/cell checks use a maximum tolerance of 1e-8 Angstrom; the packaged data match exactly.

## Figure reference and verification limits

The final figures are 147 by 76 mm with 11 pt plot text. The final preparation rendered PNGs at 400 dpi and embedded them as one opaque RGB image per PDF page. The portable exporter preserves that image without lossy compression. PDF file hashes can differ because PDF container metadata and object layout differ; embedded RGB pixels and page dimensions are the relevant comparison.

The untouched archived force metrics JSON contains an older `rendering` description mentioning vector text. The actual final figure PDF is an opaque image embed; the portable script corrects this descriptive metadata without changing any numerical or plotting formula. The original reference JSON remains intact, and verification deliberately ignores only this descriptive field. Caption labels are updated from the earlier R1/R2 names to S9/S10.

Verification checks all numerical metadata and the energy CSV against the references with `rtol=1e-12, atol=1e-10`. Pixel equality additionally checks the PNGs and is reported separately. In the environment recorded in `validation.json`, numerical reproduction passed, but regenerated PNGs were not pixel-identical to the archived figures; visible axis-label spacing/placement differs. The original figure PNGs and PDFs are preserved unchanged in `reference/`. The exact original rendering environment was not recorded, so the pinned requirements document this package's tested environment, not an asserted original environment.

For a strict image comparison, add `--require-pixel-equality`; this intentionally fails for the documented nonidentical rerender. Library/font differences can affect pixels even when numerical results match. This is a stored-data/figure reproduction check, not a new teacher-model evaluation or a validation of training provenance.
