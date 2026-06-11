# Water MLIPs — UMA-M MD-sampled configs

Trained MACE models from water MD trajectories generated with UMA-M (omol foundation model).

## Layout

- `final_models/` — final `*_stagetwo.model` files, one per training run.
  - `learnig_curve_UMA-M-omol/` — models trained at varying dataset sizes (5–400 configs).
  - `mlp-train-*/` — final models for each foundation-model teacher.

## Notes

- Total final models: 20.
- Each subdirectory contains a single `*_stagetwo.model` file.
