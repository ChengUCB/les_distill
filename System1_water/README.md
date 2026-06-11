# System 1: Bulk Liquid Water

Analysis scripts, packaged data, and training utilities for the LES distillation benchmark — bulk liquid water.

## Contents

- `water_IR/` — grouped IR spectrum plotting script and input data.
- `water_BEC/` — BEC benchmark data, plotting script, and notebook.
- `water_MLIPs_UMA-M-MD_sampled_configs/` — trained MACE models from UMA-M MD-sampled training sets.
- `water_MLIPs_RPBE-D3_sampled_configs/` — trained MACE models from RPBE-D3 sampled training sets.
- `Datasets/` — training and test xyz datasets (RPBE-D3 and UMA-M sampled).
- `MLIP_and_MD_setups/` — SLURM scripts and Python utilities for MACE training and MD runs.

## Quick start

Install the analysis dependencies:

```bash
pip install -r requirements.txt
```

Reproduce the IR figure:

```bash
python water_IR/plot_water_ir_grouped.py
```

Reproduce the BEC benchmark figures:

```bash
python water_BEC/water_BEC_benchmark.py
```

## MACE two-stage training and MD

The `MLIP_and_MD_setups/` scripts require a working [MACE](https://github.com/ACEsuit/mace) installation. See the MACE documentation for setup instructions.

```bash
# Example: run MACE two-stage training on Savio HPC
sbatch MLIP_and_MD_setups/MACELES-fit.sh

# Run MD and compute BEC current
python MLIP_and_MD_setups/md_water_IR.py --init-config <init.xyz> --model-dir <model_dir> --output-root <output>
python MLIP_and_MD_setups/BEC_part1.py --model-path <model.model> --traj-path <traj.traj>
```
