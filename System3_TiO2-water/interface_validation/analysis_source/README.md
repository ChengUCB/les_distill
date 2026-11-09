# Original analysis sources

These Python files are byte-for-byte copies from the recovered author workspace. They retain the density/orientation calculations, surface identification, intact-water reconstruction, normalization and averaging definitions. Their hashes and original relative paths are recorded in `../provenance/sources.json`.

They are a methods/provenance snapshot, not a standalone raw-data workflow: original trajectories and coordinate caches are deliberately omitted from this small backup. Do not run these archival files as a substitute for the package's `scripts/` redraw entry points. Historical local/HPC paths remain as recorded.

The runnable redraw scripts retain the original figure formatting but read the unchanged final CSV/DAT tables. No archival analysis module is imported during redraw, so SciPy, zstandard, ASE, GPUs and the original external drives are not required for figure reproduction.
