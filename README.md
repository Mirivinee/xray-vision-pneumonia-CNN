# Pneumonia Detection from Pediatric Chest X-rays

An academic TensorFlow/Keras project for comparing class-imbalance strategies in a scratch CNN and frozen/fine-tuned ImageNet transfer-learning models. This is a research prototype, not a medical diagnostic tool.

## Dataset and pipeline

The project uses the Kaggle Chest X-Ray Images (Pneumonia) dataset. The preparation notebook pools the source `train` and `val` images, creates a sorted, stratified 10% validation split with seed 42, retains the source test split, and checks exact duplicate files without deleting images. The currently verified prepared split contains 5,824 images; counts, SHA-256 file entries, and image metadata are recorded in `data/`.

The data itself is not included in this repository. Set `SPLIT_DIR` in the notebooks to the prepared `chest_xray_cnn_splits` folder. Before training, verify it against the checked-in manifest:

```powershell
python src/data/validate_dataset.py --split-dir "C:\path\to\chest_xray_cnn_splits"
```

See [docs/methodology.md](docs/methodology.md) for split provenance, preprocessing, and known limitations.

## Repository map

- `notebooks/01_data_preparation/`: prepared split and EDA notebooks.
- `notebooks/02_baseline_cnn/`: reusable imbalance helpers and three scratch-CNN experiments.
- `notebooks/03_transfer_learning/03_transfer_learning_comparison.ipynb`: primary four-configuration transfer-learning experiment.
- `notebooks/03_transfer_learning/archive/`: preserved earlier transfer notebooks; not competing final workflows.
- `notebooks/04_explainability/04_gradcam_analysis.ipynb`: Grad-CAM comparison of the four transfer-learning checkpoints.
- `models/`: existing baseline/legacy checkpoints and frozen/fine-tuned checkpoint destinations.
- `results/`: recorded baseline results, fresh experiment outputs, EDA, and Grad-CAM figures.
- `src/data/validate_dataset.py`: read-only manifest verification.
- `docs/`: methodology, results provenance, and presentation guide.

## Install

Use Python 3.12 and install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Select that Python environment as the Jupyter kernel. Run Jupyter from the repository root so notebooks resolve project-relative output paths consistently.

## Run notebooks

1. `notebooks/01_data_preparation/01_chest_xray_train_val_test_split.ipynb` downloads/prepares the data when a fresh split is needed. It writes prepared images to the KaggleHub cache, not into Git. Existing outputs are hash-checked and never overwritten; set a new empty `SPLIT_OUTPUT_DIR` to create a different split version.
2. `notebooks/01_data_preparation/02_data_eda_visualization.ipynb` summarizes image quality and class counts.
3. Execute the three baseline notebooks in order: class-weighted, oversampling, then augmentation. Each verifies the manifest, saves a run-specific model/history/configuration, selects a 90% sensitivity threshold on validation data, and evaluates on test data.
4. Execute `notebooks/03_transfer_learning/03_transfer_learning_comparison.ipynb`. It trains four DenseNet121/ResNet50 frozen/fine-tuned configurations, starting fine-tuning from each newly trained frozen checkpoint. Existing frozen models are not overwritten.
5. After all seven models train, the transfer-learning notebook first selects operating points and the preferred model using validation data only, then evaluates all seven models on the held-out test split and creates the unified comparison outputs. To run that step separately, use `python src/evaluation/evaluate_experiments.py --run-id seed42_run01`.
6. Execute `notebooks/04_explainability/04_gradcam_analysis.ipynb` after the four current-run transfer checkpoints exist. It reads validation images and writes figures under `results/gradcam/`.

To preserve the original notebook outputs, execute notebooks into run-specific result folders (run commands from the repository root):

```powershell
jupyter nbconvert --to notebook --execute notebooks/02_baseline_cnn/01_baseline_cnn_class_weighted.ipynb --output 01_baseline_cnn_class_weighted.ipynb --output-dir results/baseline_cnn/seed42_run01/notebooks
jupyter nbconvert --to notebook --execute notebooks/02_baseline_cnn/02_baseline_cnn_oversampling.ipynb --output 02_baseline_cnn_oversampling.ipynb --output-dir results/baseline_cnn/seed42_run01/notebooks
jupyter nbconvert --to notebook --execute notebooks/02_baseline_cnn/03_baseline_cnn_augmentation.ipynb --output 03_baseline_cnn_augmentation.ipynb --output-dir results/baseline_cnn/seed42_run01/notebooks
jupyter nbconvert --to notebook --execute notebooks/03_transfer_learning/03_transfer_learning_comparison.ipynb --output 03_transfer_learning_comparison.ipynb --output-dir results/transfer_learning/seed42_run01/notebooks
jupyter nbconvert --to notebook --execute notebooks/04_explainability/04_gradcam_analysis.ipynb --output 04_gradcam_analysis.ipynb --output-dir results/gradcam/seed42_run01/notebooks
```

Run focused code tests with `python -m pytest tests -q`. Fresh comparison files are written to `results/metrics/` and `results/plots/`; the comparison script is invoked by the unified evaluator.

## Existing artifacts and reproducibility

Files directly under `results/baseline_cnn/` are historical outputs from a different train/validation allocation. Fresh metrics and model outputs are stored under run-ID subfolders and are not mixed with historical results. Existing checkpoints are retained without implying equivalent provenance; `docs/experiment_results.md` distinguishes verified loadability from verified training history. New run-specific checkpoints are ignored by Git; do not commit large models or datasets without an explicit decision about artifact storage.

The original EDA outputs used a different train/validation allocation from the current prepared split. A manifest now pins the current local split for new comparisons; old baseline scores should not be treated as same-split comparisons with future results. The native-Windows TensorFlow 2.21.0 environment is CPU-only. Earlier baseline logs indicate 70-90 seconds per epoch; the complete seven-model training run is expected to take several hours, and transfer-learning runtime has not been measured. Generate the technical report only after new results and Grad-CAM examples have been reviewed.
