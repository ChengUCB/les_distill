# LES Distillation Benchmark

Data, trained models, analysis scripts, and MD/training utilities for the **local-environment spectroscopy (LES) distillation** benchmark — evaluating machine-learning interatomic potentials (MLIPs) against DFT-level vibrational spectra across three systems of increasing complexity.

## Systems

| Folder | System | Property | Method |
|--------|--------|----------|--------|
| [`System1_water/`](System1_water/) | Bulk liquid water | IR spectrum, BEC benchmark | MACE two-stage training (RPBE-D3 / UMA-M) |
| [`System2_HCl/`](System2_HCl/) | 2 M HCl solution | H₃O⁺ IR spectrum | MACE two-stage training (GGA) |
| [`System3_TiO2-water/`](System3_TiO2-water/) | Rutile TiO₂(110)/water interface | Layer-resolved IR spectrum, water density profile | CACE (SCAN); force-only fine-tuning of MACE-MP-0(L) |

## Repository layout

```
les_distill/
├── System1_water/
│   ├── Datasets/                  # train/test xyz (RPBE-D3, UMA-M)
│   ├── MLIP_and_MD_setups/        # MACE training script and MD utilities
│   ├── water_IR/                  # IR spectrum data and plotting
│   ├── water_BEC/                 # BEC benchmark data and notebook
│   ├── water_MLIPs_RPBE-D3_sampled_configs/   # trained MACE models
│   └── water_MLIPs_UMA-M-MD_sampled_configs/  # trained MACE models (learning curve)
├── System2_HCl/
│   ├── Datasets/                  # train/test xyz (2M HCl, UMA-S sampled)
│   ├── MLIP_and_MD_setups/        # MACE training script and MD/BEC utilities
│   ├── MLIPs/                     # trained MACE models (8 potentials)
│   └── HCl_solution_IR/           # H₃O⁺ IR spectra data and plotting
└── System3_TiO2-water/
    ├── MLIP_and_MD_setups/        # CACE training, NVT MD, BEC analysis, opt
    ├── MLIPs/                     # CACE models (direct fit, fine-tuned, student)
    └── surface_water_MD_results/  # IR spectra, density profiles, combined figure
```

## Requirements

Each system has its own dependency list. In general:

- **System 1 & 2**: [MACE](https://github.com/ACEsuit/mace), ASE, NumPy, Matplotlib, SciPy
- **System 3**: [CACE](https://github.com/BingqingCheng/cace), ASE, PyTorch, NumPy, Matplotlib, SciPy
