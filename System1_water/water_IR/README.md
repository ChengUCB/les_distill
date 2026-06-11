# Water IR spectra

This folder contains the script and packaged input data needed to reproduce the
grouped water IR spectrum figure.

## Files

- `plot_water_ir_grouped.py`: GitHub-ready plotting script with relative paths.
- `Water_IR_data/model_spectra/`: model `bec_dict.pkl` files.
- `Water_IR_data/experimental/water_IR.pkl`: experimental spectrum.
- `Water_IR_data/reference/`: reference MD/PIGS CSV spectra from Kovacs et al.

## Run

```bash
python plot_water_ir_grouped.py
```

The default output is `IR_plot_water_grouped.png` in this folder. A custom output
path can be supplied with:

```bash
python plot_water_ir_grouped.py --output figures/IR_plot_water_grouped.png
```
