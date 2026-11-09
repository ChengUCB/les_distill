# TiO2-water interface validation — Figures 5–7, S12, and S13

This folder preserves five final PDFs from the accepted manuscript. It is **a figure archive, not a complete reproduction package**. The figures are copied unchanged. `manifest.json` records checksums and manuscript source paths; `figure_map.csv` maps each PDF to its scientific comparison and missing material.

| Figure | Comparison | Archived PDF |
|---|---|---|
| 5 | Student water dipole orientation, grouped by DFT level; 60 ps and both interfaces | `figures/Fig5_student_water_orientation.pdf` |
| 6 | Foundation-teacher density in 1,422-atom cells versus student density in 8,748-atom cells | `figures/Fig6_foundation_mid_vs_caceles_large_density.pdf` |
| 7 | Foundation-teacher dipole orientation in 1,422-atom cells versus student orientation in 8,748-atom cells | `figures/Fig7_foundation_mid_vs_caceles_large_cos_theta.pdf` |
| S12 | Student density from 60-ps and extended 100-ps runs in 8,748-atom cells | `figures/extended_water_density.pdf` |
| S13 | Student density and dipole orientation in 1,422-atom and 8,748-atom cells | `figures/caceles_mid_vs_large_system_size.pdf` |

The interface is anatase TiO2(101). For orientation profiles, theta is measured between the intact-water molecular dipole and the outward surface normal. Figure S12 compares students distilled from MACE-MP-0(L), UMA-M(OC20), and PET-OMATPES. Figures 6, 7, and S13 compare GemNet-OC22, MACE-MH-1(OMat), and MACE-MP-0(L), or their respective students.

## Source-data status

No verified complete numerical series or final plotting scripts for these five figures were located in the two local projects searched on 2026-11-08. In particular, the search did not locate the dipole-orientation arrays, the 1,422-atom teacher/student curve sets, or the extended 100-ps density results.

The existing repository separately contains large-cell student density files under `System3_TiO2-water/surface_water_MD_results/data/water_density_profile/MD_simulations/`. These are relevant baseline candidates, but their exact run/curve mapping to Figures 6, S12, and S13 has not been established. They have not been relabeled here as verified revision source data. The general density scripts associated with the earlier combined density/IR figure do not replace the missing teacher/student and finite-size comparison scripts.

## Still needed for reproduction

- For every plotted series: a model/checkpoint identifier, atom count, production interval, trajectory or run identifier, distance grid, and density or mean cos(theta) values.
- For orientation: the intact-water definition, dipole definition, periodic-boundary treatment, surface-normal/sign convention, binning, and averaging over the two interfaces.
- For density: the water-oxygen classification, surface origin/alignment, binning, units, normalization, and any smoothing.
- The scripts that generate these arrays from trajectories and the final plotting scripts that associate each curve with the accepted panel.
- For S12: the exact original 60-ps and extended 100-ps sampling intervals and their corresponding numerical profiles.

No numerical values were inferred from filenames or digitized from the PDFs. Unavailable data are not replaced with synthetic or reconstructed curves.
