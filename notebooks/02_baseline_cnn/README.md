# Baseline CNN experiments

The three existing scratch-CNN experiments use 224×224 grayscale inputs, batch size 32, up to 10 epochs, Adam, and early stopping on validation loss (patience 3). They preserve the original imbalance methods and notebook execution outputs.

## Notebooks

- [01_baseline_cnn_class_weighted.ipynb](./01_baseline_cnn_class_weighted.ipynb): inverse-frequency weights calculated from training labels.
- [02_baseline_cnn_oversampling.ipynb](./02_baseline_cnn_oversampling.ipynb): repeats training examples to balance classes.
- [03_baseline_cnn_augmentation.ipynb](./03_baseline_cnn_augmentation.ipynb): augments training batches only.
- [00_imbalance_utilities.ipynb](./00_imbalance_utilities.ipynb): reusable class-weight, oversampling, and augmentation helpers.

Validation and test remain unbalanced and unaugmented. The test set was not used for model selection. All notebooks now default to the canonical manifest-backed split and save checkpoints under `models/baseline/`.

## Recorded results

The values in [results/baseline_cnn/](../../results/baseline_cnn/) are copied from previously saved notebook outputs, not freshly generated. The historical train/validation allocation differs by three PNEUMONIA images from the current canonical split. Do not treat these values as a same-split comparison against future transfer-learning runs.
