# LES Distillation Benchmark

Training data, training scripts, fitted machine-learning interatomic potentials (MLIPs), molecular dynamics (MD) setups, analysis scripts, and raw data for the **Latent Ewald Summation (LES) distillation** benchmark across three chemical systems.

## Systems

| Folder | System | Property | Method |
|--------|--------|----------|--------|
| [`System1_water/`](System1_water/) | Bulk liquid water | IR spectrum, BEC benchmark | MACE two-stage training |
| [`System2_HCl/`](System2_HCl/) | 2 M HCl solution | H₃O⁺ IR difference spectrum | MACE two-stage training |
| [`System3_TiO2-water/`](System3_TiO2-water/) | anatase TiO₂(101)-water interface | Surface IR spectrum, water density profile | CACE; force-only fine-tuning of MACE-MP-0(L) |

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
│   ├── MLIP_and_MD_setups/        # MACE training script and MD/BEC analyses
│   ├── MLIPs/                     # trained MACE models 
│   └── HCl_solution_IR/           # H₃O⁺ IR spectra data and plotting
└── System3_TiO2-water/
    ├── MLIP_and_MD_setups/        # CACE training, NVT MD, BEC analysis, opt
    ├── MLIPs/                     # CACE models (direct fit, fine-tuned, student)
    └── surface_water_MD_results/  # IR spectra, density profiles
```

## Requirements

Each system has its own dependency list. In general:

- **System 1 & 2**: [MACE](https://github.com/ACEsuit/mace), ASE, NumPy, Matplotlib, SciPy
- **System 3**: [CACE](https://github.com/BingqingCheng/cace), ASE, PyTorch, NumPy, Matplotlib, SciPy

## Publication figure materials

Materials associated with *Distilling latent electrostatics from foundation machine learning interatomic potentials*, [DOI: 10.1103/kgym-z2rw](https://doi.org/10.1103/kgym-z2rw):

| Figures | Directory | Contents |
|---------|-----------|----------|
| Supplemental Figs. S9/S10 | [Teacher label comparison](System3_TiO2-water/teacher_label_comparison/) | Matched configurations and labels, pairing indices, portable analysis scripts, numerical results, and figures |
| Main Figs. 5–7; Supplemental Figs. S12/S13 | [Interface validation](System3_TiO2-water/interface_validation/) | Final figure PDFs and a figure-by-figure inventory of numerical data and scripts still to be added |
| Supplemental Fig. S5 | [Computational benchmark](System1_water/computational_benchmark/) | Final figure PDF, reported benchmark protocol, and an inventory of timing/memory records and scripts still to be added |

Only the S9/S10 directory currently includes the complete inputs and scripts for
reproducing these newly archived figures. A final PDF alone is not a numerical
data archive; each directory states its current coverage explicitly.

### Downloading the S9/S10 data

The four label caches are stored using [Git LFS](https://git-lfs.com/). After
cloning the repository, retrieve the actual data files before running the
analysis:

```bash
git lfs install
git lfs pull --include="System3_TiO2-water/teacher_label_comparison/data/cache_*.npz"
```

See the [S9/S10 README](System3_TiO2-water/teacher_label_comparison/README.md)
for dependency versions, verification commands, and reproduction instructions.
