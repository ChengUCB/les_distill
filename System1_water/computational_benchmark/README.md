# Liquid-water computational benchmark — Figure S5

This folder preserves the final Figure S5 PDF from the accepted manuscript. It is **a figure archive, not a complete reproduction package**. The PDF is copied unchanged; its SHA-256 checksum and manuscript source path are recorded in `manifest.json`.

## Available

- `figures/MDspeed.pdf`: accepted Figure S5.
- `figure_map.csv`: figure description and missing material.
- `reported_protocol.json`: parameters and the single GPU-memory value explicitly reported in the Supplementary Material. These are manuscript-reported values, not recovered benchmark logs.

## Reported benchmark

The UMA-M(OMol)-derived MACELES student is compared with 12 foundation models for liquid water, from 192 to 12,288 atoms. The reported mean wall time covers 100 consecutive NVT MD steps at 300 K with a 1 fs timestep, using ASE on one NVIDIA H200 GPU (142 GB). Model loading, the initial force evaluation, and a 10 fs warm-up were excluded. Curves stop when available GPU memory is exceeded.

## Still needed for reproduction

1. Exact wall-time observations or summary tables for each model and atom count, with units and the averaging convention.
2. GPU-memory measurements and out-of-memory outcomes for each model and atom count; clarify whether each metric is allocated, reserved, or peak memory.
3. The benchmark driver, input configurations, teacher checkpoint identifiers, student checkpoint mapping, package versions, precision, and timing/synchronization settings.
4. The final Figure S5 plotting script and the table it reads.

No figure-specific timing table, complete memory log, or final S5 plotting script was located in the two local projects searched on 2026-11-08. An older notebook contains results for only UMA-S and UMA-M, without a verified link to this benchmark; those values are deliberately excluded. No curves were digitized from the PDF.
