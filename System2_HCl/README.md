# System 2: 2 M HCl Solution

Analysis scripts, packaged data, trained models, and training utilities for the Latent Ewald Summation (LES) distillation benchmark — 2 M HCl solution H₃O⁺ IR difference spectrum.

## Contents

- `HCl_solution_IR/` — pre-computed H₃O⁺ IR spectra (`.npz`), experimental reference data, and grouped plotting script.
- `MLIPs/` — eight two-stage MACE models, one per foundation model / DFT functional combination.
- `Datasets/` — training (1000 configs) and test (100 configs) xyz files sampled from UMA-S NVT MD.
- `MLIP_and_MD_setups/` — MACE two-stage training script, NVT MD runner, BEC current calculator, and RDF utility.

## Quick start

Install dependencies:

```bash
pip install mace-torch ase numpy matplotlib scipy
```

Reproduce the IR figure:

```bash
python HCl_solution_IR/ir_plot_grouped.py
```

## Models

Eight two-stage MACE models, DFT reference: GGA:

| File | Foundation model | DFT reference |
|------|-----------------|---------------|
| `2M-HCl_MACE-MH-omol_stagetwo.model` | MACE-MH (OMol) | GGA |
| `2M-HCl_MACE-MP0_stagetwo.model` | MACE-MP-0 | GGA |
| `2M-HCl_MACE_omol_stagetwo.model` | MACE (OMol) | GGA |
| `2M-HCl_UMA-M-OC20_stagetwo.model` | UMA-M (OC20) | GGA |
| `2M-HCl_UMA-S-1p2-omol_stagetwo.model` | UMA-S (OMol 1.2) | GGA |
| `2M-HCl_esen-OC25-md-dir_stagetwo.model` | eSEN-OC25-md | GGA |
| `2M-HCl_esen-OC25-sm-con_stagetwo.model` | eSEN-OC25-sm | GGA |
| `2M-HCl_orbv3-omol-cons_stagetwo.model` | Orb-v3 (OMol) | GGA |

## MACE two-stage training and MD

```bash
# Two-stage MACE training
bash MLIP_and_MD_setups/MACELES-fit.sh

# Run NVT MD
python MLIP_and_MD_setups/mace_md.py \
    --model_path MLIPs/2M-HCl_MACE-MP0_stagetwo.model \
    --init_config Datasets/train_2M_H3O_1000_UMA-S-omol-1p2.xyz \
    --temperature 300 --nsteps 400000

# Compute BEC-derived dipole current
python MLIP_and_MD_setups/bec_pro.py \
    --model-path <model.model> \
    --traj-path <traj.traj> \
    --start-frame 0 --end-frame 50000

# Compute RDF
python MLIP_and_MD_setups/rdf.py
```
