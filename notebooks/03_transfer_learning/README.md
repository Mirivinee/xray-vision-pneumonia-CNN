# Transfer-learning experiment

`03_transfer_learning_comparison.ipynb` is the single primary transfer-learning notebook. Older notebooks are preserved under `archive/` for provenance, not as alternative final workflows.

The notebook trains four configurations and saves complete Keras models under `models/frozen/` and `models/fine_tuned/` using the run ID `seed42_run01`. Existing frozen checkpoints are never overwritten. Fine-tuning starts from each corresponding frozen checkpoint from the same run. The notebook verifies the manifest before each phase, saves separate histories/configurations, and can resume already verified phases.

After the three canonical-split baseline notebooks have completed, the notebook invokes `src/evaluation/evaluate_experiments.py`. That script selects thresholds and the preferred model using validation data only, then evaluates all seven models on the held-out test split and writes the unified comparison outputs.
