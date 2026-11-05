# Class-imbalance experiments

The reusable TensorFlow/Keras helpers are in [`imbalance.ipynb`](../imbalance.ipynb).
Run its import and helper-function cells before using the examples below, or copy
those cells into the CNN teammate's notebook. They operate on datasets of
`(images, sparse_integer_labels)` with class IDs
`NORMAL=0` and `PNEUMONIA=1`, matching the sorted folder order used by the
project's prepared splits.

## What the methods do

- **Class weights** (`compute_class_weights`) counts labels in the supplied
  training dataset and returns inverse-frequency weights for
  `model.fit(class_weight=...)`. It does not duplicate or transform images.
  Every configured class must appear in training; missing classes raise a
  `ValueError` rather than producing a misleading weight.
- **Oversampling** (`oversample_training_dataset`) repeats examples from the
  minority class until each class has the same number of examples. It creates a
  new `tf.data.Dataset`, does not copy files, and defaults to the original
  majority-class count. `target_per_class` can request a larger balanced
  dataset. It uses additional steps and can increase overfitting because
  minority examples are seen repeatedly.
- **Augmentation** (`augment_training_dataset`) applies random, conservative
  rotation, zoom, and translation to training images only. It preserves image
  shape, dtype, and labels. Transform ranges can be reduced or disabled.
  Augmentation may improve generalization but changes image appearance, so
  review domain appropriateness before increasing its ranges.

Use the methods as separate experiments first. Combining them changes the
effective class emphasis and should be a separately documented experiment.

## Load the prepared folders

The prepared data is expected at
`chest_xray_cnn_splits/{train,val,test}/{NORMAL,PNEUMONIA}/`. This matches the
folder layout produced by `chest_xray_train_val_test_split.ipynb`.

```python
from pathlib import Path

import tensorflow as tf

DATA_DIR = Path("chest_xray_cnn_splits")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

tf.keras.utils.set_random_seed(SEED)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

assert train_ds.class_names == CLASS_NAMES
```

Run the imports and helper-function cells from `imbalance.ipynb` in this
notebook's kernel before running the examples below.

`image_dataset_from_directory` returns finite batches with integer labels by
default; the explicit arguments above pin the class mapping and label format.
The weighting and oversampling helpers make a pass over the finite training
dataset to count its labels. Do not pass an infinite/repeated dataset to them.

## Run each experiment separately

### 1. Class-weighted loss

Pass weights calculated from **training labels only** to the CNN's `fit` call:

```python
class_weights = compute_class_weights(train_ds, num_classes=2)

# model is supplied by the CNN implementation; see the template below.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    class_weight=class_weights,
    epochs=EPOCHS,
)
```

### 2. Oversampling

This repeats minority examples in a new training dataset and re-batches it:

```python
balanced_train_ds = oversample_training_dataset(
    train_ds,
    num_classes=2,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

history = model.fit(
    balanced_train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)
```

To request more than the default majority-class count per class, set
`target_per_class` to an integer at least as large as the original largest
class. Do not pass `balanced_train_ds` to validation or testing.

### 3. Data augmentation

Augment a training dataset without changing the validation or test datasets:

```python
augmented_train_ds = augment_training_dataset(
    train_ds,
    rotation_factor=0.02,
    zoom_factor=0.05,
    translation_factor=0.02,
    seed=SEED,
)

history = model.fit(
    augmented_train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)
```

Each transform can be turned off by setting its factor to `0`. Rotation is a
fraction of a full turn; zoom and translation factors are fractions of the
image dimensions. These helpers do not normalize image pixel values; apply the
same preprocessing expected by the model.

## CNN integration template

> **Model-specific template — not runnable until the CNN teammate supplies the
> model, output shape, loss, optimizer, and training configuration.** There is
> currently no CNN implementation or training loop in this repository. The
> values below are placeholders, not recommendations for the model's output or
> loss.

```python
# TEMPLATE ONLY: replace all placeholders with the CNN implementation's
# compatible values. Compile exactly once for each experiment configuration.
model = build_cnn_model()  # supplied by the CNN teammate
model.compile(
    optimizer=OPTIMIZER,       # set for the selected CNN
    loss=MODEL_COMPATIBLE_LOSS,  # depends on output shape and label format
    metrics=MODEL_METRICS,
)

# Select exactly one training dataset/weight configuration for an experiment:
experiment_train_ds = train_ds
experiment_class_weights = None

# Class-weight experiment:
# experiment_class_weights = compute_class_weights(train_ds, num_classes=2)

# OR oversampling experiment:
# experiment_train_ds = oversample_training_dataset(
#     train_ds, num_classes=2, batch_size=BATCH_SIZE, seed=SEED
# )

# OR augmentation experiment:
# experiment_train_ds = augment_training_dataset(train_ds, seed=SEED)

history = model.fit(
    experiment_train_ds,
    validation_data=val_ds,
    class_weight=experiment_class_weights,
    epochs=EPOCHS,  # supplied by the CNN teammate
)
test_metrics = model.evaluate(test_ds)  # evaluate the held-out test split once
```

The model must accept the prepared image dimensions and integer labels, or the
CNN teammate must explicitly adapt the model/loss/label handling. In particular,
the appropriate output layer and loss cannot be selected until the model's
output shape is known.

## Compare experiments

Run the same CNN configuration and training protocol in three separate runs,
selecting one method at a time above. Keep the random seed, epoch budget,
preprocessing, and other hyperparameters fixed where possible. Record training
and validation metrics for each run, then use the held-out test set for final
comparison only; do not tune based on test results.

**Validation and test data must remain unbalanced and unaugmented.** Never use
class weights, oversampling, or augmentation to alter `val_ds` or `test_ds`.
