# Methodology and dataset provenance

## Objective and labels

The project classifies pediatric chest X-rays into `NORMAL` and `PNEUMONIA`. Directory loading sorts class names, so the verified mapping is `NORMAL=0`, `PNEUMONIA=1`. The system is an academic prototype, not a diagnostic tool.

## Split construction and discrepancy

The preparation notebook downloads `paultimothymooney/chest-xray-pneumonia`, pools source `train` and `val`, uses scikit-learn `train_test_split(test_size=0.10, random_state=42, stratify=labels)`, and keeps source `test` separate. The updated notebook sorts collected paths before splitting and checks SHA-256 duplicates across the prepared splits. It does not delete images; if duplicates are found, it stops for manual review. Existing output folders are hash-compared and preserved rather than overwritten. The notebook does not group by patient.

The previously saved EDA/baseline outputs report 4,688 train, 518 validation, and 618 test images. The current prepared split contains 4,685 train, 521 validation, and 618 test images. Both totals are 5,824; train+validation have the same class totals (`NORMAL=1,348`, `PNEUMONIA=3,858`), but three PNEUMONIA cases differ in their train/validation allocation. I replayed the seed-42 split against the locally cached source data and compared per-image SHA-256 values: the sorted-source split exactly matches all 5,824 current prepared images and assignments. Historical outputs have no per-image manifest, so their exact membership cannot be compared. The original code did not sort filesystem traversal; source ordering is a plausible cause of the historical three-image shift, but the original input order is unavailable and the cause cannot be proved.

The current local prepared split is the canonical split for new experiments. Its file paths, SHA-256 values, labels, dimensions, and modes are stored in `data/dataset_manifest.csv`; aggregate metadata and the manifest hash are in `data/dataset_manifest_summary.json`. Verify the actual dataset before training:

```powershell
python src/data/validate_dataset.py --split-dir "C:\path\to\chest_xray_cnn_splits"
```

The current manifest records 5,824 readable images and zero exact duplicate hashes, including across splits. The separate test support is 231 NORMAL and 387 PNEUMONIA. The source test count and support match the historical outputs, but matching counts do not prove the same test-image membership. Do not compare historical baseline scores to new transfer-learning scores as a controlled same-split comparison.

To reproduce the new comparisons, preserve the prepared directory matching the manifest and verify it before training. The preparation notebook now sorts input paths and writes a new manifest after creating a split. Its source-data membership was verified against the current local source cache; other source versions should be treated as new split versions and must not silently replace this manifest. Do not retune on test data.

## Image handling

The scratch CNN notebooks resize to 224×224 grayscale and rescale pixels by 1/255. The transfer-learning notebook resizes to 224×224 RGB, uses ImageNet initialization and the matching DenseNet121 or ResNet50 `preprocess_input`, and applies random flip, small rotation, and zoom only during training. Validation and test images are not augmented.

The class-weighted baseline computes inverse-frequency weights on training labels. The oversampling baseline repeats minority-class training examples; the augmentation baseline transforms training images. All validation and test splits remain unbalanced and unaugmented. New comparisons select the operating threshold on validation scores by maximizing precision subject to at least 90% PNEUMONIA sensitivity. The selected threshold is fixed before held-out test evaluation.

## Limitations

The dataset is imbalanced and may contain label noise, source-specific artifacts, and demographic or acquisition biases. There is no patient-level grouping in the split code, so patient-level leakage cannot be ruled out. Exact byte hashes detect identical files, not near-duplicates or images of the same patient. Grad-CAM is a qualitative visualization; it neither proves clinical reasoning nor establishes that an artifact caused a prediction. Evaluation on this dataset does not establish clinical validity or external generalization.

The reproducibility environment is pinned in `requirements.txt` for Python 3.12 and TensorFlow 2.21.0. The current native-Windows TensorFlow runtime exposes only a CPU device. Runtime estimates are hardware-dependent; previous baseline logs indicate roughly 70-90 seconds per epoch, and no transfer-learning timing is available before a new training run.
