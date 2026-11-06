# Baseline CNN experiments

The three existing scratch-CNN experiments use 224×224 grayscale inputs, batch size 32, up to 10 epochs, Adam, and early stopping on validation loss (patience 3). They preserve the original imbalance methods and notebook execution outputs.

## Notebooks

- [01_baseline_cnn_class_weighted.ipynb](./01_baseline_cnn_class_weighted.ipynb): inverse-frequency weights calculated from training labels.
- [02_baseline_cnn_oversampling.ipynb](./02_baseline_cnn_oversampling.ipynb): repeats training examples to balance classes.
- [03_baseline_cnn_augmentation.ipynb](./03_baseline_cnn_augmentation.ipynb): augments training batches only.
- [00_imbalance_utilities.ipynb](./00_imbalance_utilities.ipynb): reusable class-weight, oversampling, and augmentation helpers.

Validation remains unbalanced and unaugmented. Training notebooks do not load or evaluate the held-out test split. Each checkpoint is saved under a run-specific filename in `models/baseline/`; run histories and configurations are stored under `results/baseline_cnn/seed42_run01/`.

After the three baselines and four transfer models are trained, the unified evaluator selects a preferred model and validation operating thresholds, then evaluates all seven configurations on test data.

## Recorded results

The values in [results/baseline_cnn/](../../results/baseline_cnn/) are copied from previously saved notebook outputs, not freshly generated. The historical train/validation allocation differs by three PNEUMONIA images from the current canonical split. Do not treat these values as a same-split comparison against future transfer-learning runs.
