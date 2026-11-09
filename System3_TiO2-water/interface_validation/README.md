# TiO2–water interface validation: Figures 5–7, S12 and S13

This package redraws all 41 plotted curves from the recovered final numerical tables. It also preserves the original analysis source code as a methods record. The five accepted PDFs already in `figures/` are retained; no duplicate figure copies, raw trajectories, model weights, or position caches are added.

## Reproduce the figures

Use Python 3.10 or newer, NumPy and Matplotlib:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
python scripts/verify_reproduction.py
```

The first command after installation writes PDFs and PNGs to `reproduced/`. Use `--output-dir /your/output/folder` to choose another destination. Each `scripts/plot_*.py` also runs independently. All inputs resolve relative to this package, so the working directory does not matter.

| Figure | Numerical input in `data/` | Redraw script |
|---|---|---|
| 5 | `all_caceles_student_water_orientation.csv` | `plot_fig5.py` |
| 6 | `foundation_mid_vs_caceles_large_density.dat` | `plot_fig6.py` |
| 7 | `foundation_mid_vs_caceles_large_cos_theta.dat` | `plot_fig7.py` |
| S12 | `old_new_large_water_density.dat` | `plot_s12.py` |
| S13 | `caceles_mid_vs_large_density.dat` and `caceles_mid_vs_large_cos_theta.dat` | `plot_s13.py` |

The Fig. 5 summary CSV preserves model identifiers, original trajectory paths, analyzed-frame counts and averaging settings. DAT headers name every model, grid and observable. All supplied tables are unchanged copies of the final author exports; values were not digitized or inferred from the PDFs. The exported values have eight decimal places.

## What is reproduced

The runnable scripts retain the original panel layout, model ordering, colors, line styles, limits, labels and legends. Their inputs are the final tables instead of trajectory analysis or cached coordinates. There is no new smoothing, interpolation, normalization, rebinning, or model fitting. NaN bins are retained, so lines do not bridge excluded regions.

This is **final-curve and figure reproduction**, not a complete raw-MD reanalysis package. `analysis_source/` preserves the original scientific calculations and required local helper modules unchanged. Those archival files require their original raw trajectories/caches and are not executed by the redraw commands. The source-to-file mapping and checksums are in `provenance/sources.json`.

The interface is anatase TiO2(101). For orientation the molecular direction is O toward the H midpoint, projected on the **local inward normal** from the nearest surface into the water film (+z at the lower interface and −z at the upper). This corrects the earlier archive README's “outward” wording to match the recovered code. Figure-specific molecule selection and interface averaging remain in the archived methods; the redraw scripts use the already-computed profiles unchanged.

## Validation and provenance

The original recovered PDFs were SHA-256 identical to all five accepted PDFs already archived here. The verified source directory was `MISC/Rev_distill/`, recovered on 2026-11-08. S12's original file was `Sensitivity_MDduration/old_new_large_water_density_horizontal.pdf`, identical to the archived `figures/extended_water_density.pdf`.

`verify_reproduction.py` verifies all 17 copied source/data payload hashes and checks the x/y arrays, NaN masks and order of all **41 curves** against their respective tables. All five redraw commands completed under NumPy 2.2.6 and Matplotlib 3.10.3; see `validation.json`. Other fonts/Matplotlib versions can change typography or PDF metadata. Byte-identical regenerated PDFs are not promised.

Timing labels and curves are preserved as received. `provenance/README.md` records the recovered analyzed-frame counts, including differences between nominal timing labels and some underlying runs; this backup does not alter the manuscript or silently choose another time window.
