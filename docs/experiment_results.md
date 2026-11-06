# Experiment results and provenance

## Recorded baseline results

These metrics are transcribed from the saved test outputs in the baseline notebooks and the baseline README. They were **not rerun** during reorganization. All three reports show 618 test images (231 NORMAL, 387 PNEUMONIA), but they were produced with an earlier train/validation allocation than the currently manifested split.

| Method | Test loss | Accuracy | Macro F1 | NORMAL recall | PNEUMONIA recall |
|---|---:|---:|---:|---:|---:|
| Class-weighted CNN | 0.5102 | 0.7735 | 0.7378 | 0.5411 | 0.9121 |
| Oversampling CNN | 1.2556 | 0.6262 | 0.3851 | 0.0000 | 1.0000 |
| Training augmentation CNN | 0.5662 | 0.7201 | 0.6251 | 0.2900 | 0.9767 |

The oversampling output predicts every test case as PNEUMONIA. Accuracy therefore hides a complete failure to identify NORMAL cases. See the notebook outputs and `results/baseline_cnn/` for the recorded reports, confusion matrices, and training plots.

## Transfer learning

No fresh canonical-split baseline or transfer-learning training/test evaluation has been run yet. The baseline notebooks and the consolidated transfer-learning notebook write run-specific checkpoints, histories, and configurations. After all seven models are trained, `python src/evaluation/evaluate_experiments.py --run-id seed42_run01` selects the model and each operating threshold using validation data only, then evaluates all seven models on test data. Threshold selection maximizes precision subject to at least 90% PNEUMONIA sensitivity. The evaluator writes per-model reports/plots plus one unified comparison CSV, Markdown table, and plot.

The existing frozen DenseNet121 and ResNet50 files load in the current TensorFlow environment when their architecture-specific `preprocess_input` custom object is supplied; both have zero trainable backbone weights. Their earlier training runs are not present in the notebooks' execution outputs, so their training history and metrics are unverified. The consolidated experiment saves to distinct run-ID filenames, leaving those existing frozen files untouched. The older `models/transfer_learning/densenet121_pneumonia_legacy.keras` also loads, but its backbone is trainable; its exact training history/provenance is unknown. It is not treated as a verified fine-tuned result.

## Checkpoint policy

The repository has no Git LFS attributes configured. Existing checkpoints are preserved; new run-ID baseline, frozen, and fine-tuned checkpoints are excluded by `.gitignore`. Datasets and machine-generated model files should remain outside commits unless the project explicitly chooses Git LFS or an external artifact store.
