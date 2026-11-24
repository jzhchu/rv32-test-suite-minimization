# Supplementary analysis

This directory contains the controls, rounding-budget ablation, numeric witnesses,
and figures used in the report's final comparison.

Run these commands from the repository root after installing `requirements.txt`:

```bash
python supplement/audit_analysis.py
python supplement/additional_analysis.py --reuse-ablation
```

Both scripts read the archived primary `results/` directory by default. Their
`--results` and `--output` options select a fresh full-grid run and a separate
destination. The `--reuse-ablation` option expects an existing
`analysis/rounding_ablation.csv` in the selected destination.

The saved evidence includes:

- 110 random suites with identical reverse deletion, for a fair quality control.
- 120 checks of detection retained for 35 variants excluded from each smaller
  optimization target set; these variants come from the same catalog.
- 110 LP primal/dual numerical checks, with vectors in `audit_lp_witnesses.npz`.
- 280 LP rounding calls on seven fixed matrices, four candidate budgets, and ten
  algorithm seeds. Budget one is deterministic near-one selection with repair.
- Timing stability and time-limited IP bounds derived from the frozen primary
  records. Their repetitions are not new independent datasets.

`main_scaling` and `comparative_analysis` are the report's two figures.
`stress_tradeoff` and `process_ablation` retain separate views of their panels.

The scripts preserve primary data. Running the ablation again produces new
supplementary wall-clock timings; use `--reuse-ablation` to redraw from the
archived measurements.
