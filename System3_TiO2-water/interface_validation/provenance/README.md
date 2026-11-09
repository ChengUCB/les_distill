# Provenance and sampling notes

Recovered source root: `MISC/Rev_distill/`. The five original outputs matched the accepted PDF SHA-256 values in `../manifest.json`. The archived numerical tables are byte-for-byte copies; no values were recovered from images. Runnable scripts are adaptations for final-table input, while `analysis_source/` contains unchanged original method code.

The underlying analysis generally keeps the final 75% of saved snapshots. The recovered Fig. 5 summary records 300 analyzed snapshots for most models, 285 for eSEN-direct, and 375 for MACE-H1-r2scan and PET-OMATPES. Its complete model/trajectory mapping remains in the supplied summary CSV.

For S12, the previous-run caches contain 300 retained snapshots for MP0 and UMA-M and 375 for PET; all new runs contain 150. The new 8748-atom runs used 1 fs steps, 50 ps warmup and 100 ps production, saved every 0.5 ps. The retained 150-snapshot span is 74.5 ps. The original legend and DAT header label previous production as 60 ps, although the inspected PET previous run is 75 ps. This package preserves that existing label and curve rather than silently changing the scientific record.

For the three 1422-atom student runs used by S13, the cached positions contain 300 retained of 400 saved snapshots (200 ps production, final 75%). Teacher caches used by Figs. 6/7 contain 422/563 retained/source snapshots for GemNet, 504/672 for MH and 562/750 for MP0. The corresponding large-cell student caches contain 300/400 snapshots. These exact source selections should be retained when comparing a manuscript's timing description; a later longer raw run is not interchangeable with the original plotted average.

The original Fig. 6/7 DAT exports and S12 DAT are also byte-identical to their nested analysis-result copies. Fig. 5 means and S12 densities were compared to the saved numerical caches: differences are at most 5e-9, the expected eight-decimal export rounding. Those redundant caches are not included.
