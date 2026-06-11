# Water BEC benchmark

This folder packages the water BEC benchmark data used by the cleaned notebook
and script.

## Files

- `data/h2o_bec.xyz`: DFT reference BEC data.
- `data/bec_*.xyz`: model-predicted BEC, energy, and force data used for the
  parity and kPCA benchmark plots.
- `water_BEC_benchmark.py`: command-line script for reproducing the benchmark
  figures.
- `water_BEC_benchmark.ipynb`: notebook entry point using the same relative-path
  script.

## Run

```bash
python water_BEC_benchmark.py
```

The default outputs are written to `figures/`.
