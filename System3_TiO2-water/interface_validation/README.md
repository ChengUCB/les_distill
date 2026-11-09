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
