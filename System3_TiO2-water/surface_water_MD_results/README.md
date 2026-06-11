# Surface Water MD Results — TiO₂/Water Interface

Combined figure of water density profiles and IR spectra at the rutile TiO₂(110)/water interface, benchmarking multiple machine-learning interatomic potentials (MLIPs) against DFT-level references.

## Output

Running `finalplot_combined_density_ir.py` produces:

- `TiO2_density_IR_combined.pdf`
- `TiO2_density_IR_combined.png`

### Figure layout (3 × 2 grid)

|  | Density profile | IR spectrum |
|--|--|--|
| **PBE** | Distance from TiO₂ surface vs. density | Wavenumber vs. normalized intensity |
| **RPBE(-D3)** | same | same |
| **meta-GGA (SCAN)** | same (thicker border) | same (thicker border) |

The right-column top cell holds a compact table legend distinguishing interfacial water (solid line) from bulk water (dashed line), and simulation (Sim) from experiment (Exp).

## Models

| Functional | Model | Label |
|--|--|--|
| PBE | MACE-MP-0(L) | `MACE_MP0a_L` |
| PBE | MACE-MH-1(OMat) | `MACE-MH-omat` |
| PBE | GemNet-OC22 | `OC22` |
| RPBE | UMA-M(OC20) | `UMA-M-oc20` |
| RPBE | eSEN-OC25-sm †| `esen-sm-con` |
| RPBE | eSEN-OC25-md †| `esen-direct` |
| SCAN | CACELES (5.5 Å) | `DFT` |
| SCAN | CACELES (6 Å, T=1) | `DFT_cut6` / `DFT_CUT6` |
| SCAN | MACE-MP-0(L) → 10% SCAN | `frac10` |
| SCAN | MACE-MP-0(L) → 50% SCAN | `frac50` |
| SCAN | MACE-MH-1(r²SCAN) | `MACE-H1-r2scan` |
| SCAN | PET-OMATPES (r²SCAN) | `PET-r2scan` |

† with D3 dispersion correction.

References shown in density panels: DPLR (Zhang et al., 2025).  
References shown in IR panels: experimental bulk water (shaded), experimental interfacial water (dotted), DPLR interfacial IR (SCAN panel only).

## Data layout

```
data/
├── water_density_profile/
│   └── MD_simulations/
│       ├── <model>/TiO2-water-density_profile.txt   # columns: z, ..., density(g/mL)
│       └── Finetune/
│           ├── MP0a_L_fine-tune_Fonly_frac10/
│           └── MP0a_L_fine-tune_Fonly_frac50/
├── IR_spectra/
│   └── <model_key>_{1st,Bulk}.npz    # keys: omega [cm⁻¹], intensity [a.u.]
└── reference/
    ├── JCP_TiO2_water_den_profile.csv        # DPLR density reference
    ├── JCP_DPLR_water_at_interface_fig2.csv  # DPLR interfacial IR
    ├── JCP_EXP_water_at_interface_fig2.csv   # experimental interfacial IR
    └── water_IR.pkl                          # experimental bulk IR
```

`IR_spectra/*.npz` contain pre-computed power spectra (FFT of dipole-current autocorrelation) for interfacial (`1st`) and bulk (`Bulk`) water layers.

## Usage

```bash
python finalplot_combined_density_ir.py
```

Requires: `numpy`, `matplotlib`, `scipy`.

## Alignment

All density profiles are x-shifted so that the first crossing of 0.1 g/mL aligns with the CACELES (5.5 Å) reference curve.  
IR spectra are normalized to their peak value in the O–H stretch region (2800–3800 cm⁻¹).
