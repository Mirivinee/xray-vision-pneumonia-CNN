# Evaluation workflow

`binary_metrics.py` contains the shared validation-sensitivity threshold selector, metric calculation, and result-artifact writer. `evaluate_experiments.py` loads the three fresh baseline and four transfer-learning checkpoints, selects each threshold and the preferred model using validation data only, and touches the held-out test split only after those choices are fixed. `compare_experiments.py` writes the consolidated CSV, Markdown table, and plot after the seven per-model results exist.

Run the workflow after training all seven models:

```powershell
python src/evaluation/evaluate_experiments.py --run-id seed42_run01
```

The evaluator verifies the dataset manifest before loading splits and keeps run-specific outputs separate from historical baseline artifacts.
