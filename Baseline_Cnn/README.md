# Baseline CNN results

This folder contains three baseline CNN experiments on the prepared chest X-ray train/validation/test splits. Each uses 224×224 grayscale images, batch size 32, a maximum of 10 epochs, and early stopping on validation loss (patience 3). The test set contains 618 images: 231 `NORMAL` and 387 `PNEUMONIA`.

## Recorded results

| Imbalance method | Best logged validation accuracy | Test accuracy | Test macro F1 | Test recall: NORMAL | Test recall: PNEUMONIA |
|---|---:|---:|---:|---:|---:|
| Class-weighted loss | 0.8822 (epoch 3) | 0.7735 | 0.7378 | 0.5411 | 0.9121 |
| Oversampling | 0.7394 (all 5 logged epochs) | 0.6262 | 0.3851 | 0.0000 | 1.0000 |
| Training-only augmentation | 0.8822 (epoch 7) | 0.7201 | 0.6251 | 0.2900 | 0.9767 |

These are the outputs recorded in the notebooks and are a short snapshot, not a guarantee of results on a new run. The notebooks may produce different results if rerun, depending on the data split, software, hardware, and random operations.

## Short interpretation

- **Class weighting** had the strongest recorded test performance of these three runs, with 0.7735 accuracy and a more balanced class recall than the other methods. It still missed 45.89% of `NORMAL` images.
- **Oversampling** did not work well in this recorded run: validation accuracy stayed at 0.7394, and the test model predicted every image as `PNEUMONIA`. Its 0.6262 accuracy matches the test-set `PNEUMONIA` proportion, while `NORMAL` recall and macro F1 were zero.
- **Augmentation** achieved 0.7201 test accuracy and high `PNEUMONIA` recall, but only 0.2900 `NORMAL` recall. It therefore still missed most normal images.

Accuracy alone hides the minority-class failure in the oversampling run. Compare per-class recall, macro F1, and the confusion matrices when assessing these experiments. In particular, investigate the oversampling run's training and validation predictions before using it.

## Notebooks

- [`baseline_cnn_class_weighted.ipynb`](./baseline_cnn_class_weighted.ipynb) — inverse-frequency class weights computed from training labels only.
- [`baseline_cnn_oversampling.ipynb`](./baseline_cnn_oversampling.ipynb) — repeats minority-class training examples; validation and test remain unchanged.
- [`baseline_cnn_augmentation.ipynb`](./baseline_cnn_augmentation.ipynb) — applies random rotation, zoom, and translation to training batches only.

Run the data-preparation notebook first to generate the split folders. Each experiment evaluates on the same held-out test set; do not use test results to tune model settings.
