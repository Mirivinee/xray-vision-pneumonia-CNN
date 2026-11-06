# Transfer-learning experiment

`03_transfer_learning_comparison.ipynb` is the single primary transfer-learning notebook. Older notebooks are preserved under `archive/` for provenance, not as alternative final workflows.

The notebook trains four configurations and saves complete Keras models under `models/frozen/` and `models/fine_tuned/`. It uses distinct `*_comparison_frozen.keras` and `*_comparison_fine_tuned.keras` names so existing frozen checkpoints are never overwritten. Before executing training, verify the prepared dataset with `src/data/validate_dataset.py`.
