# Presentation guide

## Recommended execution order

1. Open `notebooks/01_data_preparation/02_data_eda_visualization.ipynb` to discuss class balance, image dimensions, and image quality. Use the accompanying saved output as historical EDA and point out that it reflects the earlier 4,688/518 split.
2. Show `notebooks/01_data_preparation/01_chest_xray_train_val_test_split.ipynb` for stratification, the seed, exact duplicate checking, and the held-out test split. Existing split folders are preserved and checked rather than overwritten. Run `src/data/validate_dataset.py` on the chosen prepared directory to show that it matches the current manifest.
3. Present the historical baseline experiments in `notebooks/02_baseline_cnn/`, emphasizing the oversampling run's NORMAL recall of zero. Then compare fresh canonical-split results from `results/metrics/model_comparison_seed42_run01.md`; historical outputs and new results are stored separately.
4. Open `notebooks/03_transfer_learning/03_transfer_learning_comparison.ipynb`. Explain ImageNet initialization, frozen feature extraction, and fine-tuning from each newly trained frozen checkpoint. A full run can take several hours on the currently available CPU-only Windows runtime.
5. Open `notebooks/04_explainability/04_gradcam_analysis.ipynb` after the four current-run checkpoints exist. It uses validation-selected thresholds and generates correctly classified NORMAL and PNEUMONIA examples plus an error example when available for each model. Discuss lung-region versus border/text activation cautiously; Grad-CAM is qualitative, not proof of clinical reasoning.

## Before showing results

- Distinguish the old baseline outputs from fresh runs.
- Confirm which split manifest generated each result.
- Do not tune a threshold or select a model from test data.
- Do not present this prototype as a diagnostic tool.
- New evaluation uses a 90% minimum validation sensitivity target for PNEUMONIA and maximizes precision among thresholds that meet it.
- Generate the technical PDF report only after experiment metrics and Grad-CAM examples are available and reviewed.

The transfer-learning execution duration is hardware-dependent and has not been measured in this reorganized workflow.
