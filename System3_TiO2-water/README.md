# System 3: TiO₂(110)-Water Interface

Analysis scripts, trained models, and training/MD utilities for the Latent Ewald Summation (LES) distillation benchmark — TiO₂(110)-water interface. Models are evaluated on surface IR spectra and water density profiles against a CACE-SCAN reference.

## Contents

- `surface_water_MD_results/` — pre-computed IR spectra and density profiles for all models, combined figure script, and reference data.
- `MLIPs/` — CACE models organized by training strategy.
- `MLIP_and_MD_setups/` — CACE training, NVT MD, geometry optimization, BEC current analysis, and density profiling scripts.

## Quick start

Install dependencies:

```bash
pip install cace-torch ase torch numpy matplotlib scipy
```

Reproduce the combined density/IR figure:

```bash
python surface_water_MD_results/finalplot_combined_density_ir.py
```

## Models

| Directory | Description |
|-----------|-------------|
| `MLIPs/Direct_fit/` | CACE model trained directly on SCAN DFT data (`best_model_DFT_SCAN.pth`) |
| `MLIPs/Fine-tuned_models/` | MACE-MP-0(L) fine-tuned on SCAN forces (10 % and 50 % data fractions) |
| `MLIPs/Student_models/` | Foundation MLIPs (MACE, UMA, eSEN, PET, GemNet-OC22) used as student models |

## CACE training and MD

```bash
# Compute per-element reference energies
python MLIP_and_MD_setups/calc_e0.py \
    --train-path train-TiO2-water-DFT.xyz

# Train CACE model with Ewald long-range electrostatics
python MLIP_and_MD_setups/CACELES-train.py

# Geometry optimisation of the initial structure
python MLIP_and_MD_setups/opt2.py \
    --input tio2_nacl.xyz \
    --model-path MLIPs/Direct_fit/best_model_DFT_SCAN.pth

# Run NVT MD (equilibration + production)
python MLIP_and_MD_setups/cace_md_NVT.py \
    --model-path MLIPs/Direct_fit/best_model_DFT_SCAN.pth \
    --init-config MLIP_and_MD_setups/ini.xyz \
    --temperature 330 --nsteps 200000

# Compute water density profile and detect solvation-layer boundaries
python MLIP_and_MD_setups/rho.py --traj md_out.xyz

# Compute layer-resolved BEC dipole currents
python MLIP_and_MD_setups/BEC_part1.py \
    --model-path MLIPs/Direct_fit/best_model_DFT_SCAN.pth \
    --traj-path md_TiO2-water.traj \
    --start-frame 0 --end-frame 50000
```

## Data layout

```
surface_water_MD_results/
├── data/
│   ├── IR_spectra/
│   │   └── <model>_{1st,Bulk}.npz      # interfacial and bulk IR spectra
│   ├── water_density_profile/
│   │   └── MD_simulations/<model>/     # density profile txt files
│   └── reference/                      # experimental and DPLR references
└── finalplot_combined_density_ir.py
```
