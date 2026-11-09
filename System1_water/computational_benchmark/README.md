# Liquid-water computational benchmark — Figure S5

This folder contains the final plotting source, its original numerical table, and the 169 original case records selected for Figure S5 (139 successful points, 5 failed cases, and 25 larger cases skipped after an out-of-memory failure). The only figure change is the authorized MACE-MH-1 legend correction from OMat to OMol; timing data are unchanged.

## Reproduce the figure

From this folder, install `requirements-plot.txt` and run:

```bash
python plot_inference_time.py
```

The original script reads `speed_tests_combined.csv` beside itself and writes `fig11_inference_time.pdf` and `.png`. The published figure is stored only once, as `figures/MDspeed.pdf`. Matplotlib 3.10.3 reproduces the 105 mm layout. The CSV is preserved byte-for-byte, including 26 unused FP64 MACELES/CACELES rows; the original script selects the 13 plotted models.

## Measurements and protocol

`records/` preserves the selected timing, peak allocated/reserved GPU-memory measurements, failures, software versions, precision, and historical driver hashes. `reported_protocol.json` summarizes those records. Personal path prefixes and hostnames were redacted; numeric values and model IDs were not changed. `provenance.json` records original and published hashes.

The plotted student is **FP32 MACELES distilled from conservative Orb-v3 OMol**, trained on teacher-relabeled RPBE-D3 water configurations. The records specify **100 timed NVT Nose–Hoover MD steps at 300 K, a 1 fs timestep, and 5 excluded warmup steps** on an NVIDIA H200. Wall time includes the ASE integrator and calculator work. Memory values are in **GiB**. At 12,288 atoms the student uses 15.2262 GiB peak allocated and 41.2051 GiB peak reserved memory.

These source-backed details supersede the previous manuscript-transcribed metadata (UMA teacher, 10 fs warmup, and generic GB occupancy). They do not imply a benchmark rerun or further manuscript edits.

## Recovered benchmark implementation

`benchmark/benchmark_md.py`, `run_one_h200.sh`, and the 192-atom `ini.xyz` preserve the recovered implementation with only resource/interpreter paths adapted. The existing repository student checkpoint is reused; no weights or trajectories are duplicated. For an H200 run with the matching model environment, for example:

```bash
BENCHMARK_PYTHON=python bash benchmark/run_one_h200.sh maceles-orbmol-cons-fp32 1x1x1
```

External teacher checkpoints are not bundled; their filenames/API identifiers and recorded package versions are in the driver and protocol. The GemNet branch also needs the supplied recovered `benchmark/gemnet_compatibility/graph_patch.py`.

The recovered driver corresponds to historical hash `f14e1649…`, used by the LES runs. The exact older `bd605fe3…` foundation/Orb driver was not recovered. Detailed warmup-reset, synchronization and memory-reset semantics are therefore directly verified only for the recovered LES implementation. The historical student-weight hash and GemNet helper hash were not logged; their unchanged historical identity is unknown. `validation.json` records the data and figure checks performed for this archive.
