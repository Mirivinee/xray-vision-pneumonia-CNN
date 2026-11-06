# X-Ray Vision: Pneumonia Detection from Pediatric Chest Radiographs

Atria University · Neural Network and Convolutional Neural Network Sprint · Group 2

> **Not for clinical use.** This is an academic exercise on a public research dataset. The models are not validated for diagnosis, triage or screening of real patients.

## Overview

CNN classifiers (scratch and transfer learning) that label a pediatric chest X-ray as NORMAL or PNEUMONIA, evaluated the way a clinician would: sensitivity first, a threshold chosen on validation data, error analysis, and explainability checks.

Main design decisions:

- Patient-grouped, stratified train/validation split built from filenames, with exact-duplicate removal and a committed manifest (`data/splits/manifest.csv`).
- The official test set stays intact (duplicates are removed from the training pool, not from test) and is evaluated only after the model and threshold are frozen. Every test evaluation is logged in `results/test_eval_log.json`.
- One shared training recipe for scratch and transfer models; imbalance strategies (class weights, fixed oversampling, focal loss, augmentation) compared over several seeds on validation.
- Frozen-head and fine-tuned variants for DenseNet121, ResNet50 and EfficientNet-B0, plus fine-tuning and augmentation ablations.
- Threshold chosen on validation as the highest cut-off that keeps sensitivity at or above 97%, with ROC/PR curves, bootstrap confidence intervals, a screening table at several prevalences, calibration, test-time augmentation, selective prediction and an ensemble.
- Grad-CAM, Grad-CAM++ and occlusion maps on correct and incorrect cases (including false negatives), quantified region attention, masking probes and a randomisation sanity check.

## Results

Run the notebooks, then `python scripts/make_results_table.py` writes `results/summary.md`. Paste the final tables here.

| Model | Threshold | Sensitivity | Specificity | ROC-AUC | Missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|
| _fill from `results/summary.md`_ | | | | | | |

## Repository layout

```
README.md
requirements.txt
.gitignore
data/               download instructions + splits/manifest.csv
notebooks/
  01_eda.ipynb                          manifest, split audit, EDA, shortcut checks
  02_baseline_cnn.ipynb                 scratch CNN, imbalance strategies, ablations
  03_transfer_learning.ipynb            frozen vs fine-tuned, ablations
  04_evaluation_and_explainability.ipynb  validation selection, threshold, test, errors, Grad-CAM
src/
  config.py  utils.py  dataset.py  eda.py  models.py  train.py  evaluate.py  gradcam.py
scripts/            smoke_test.py, run_all.sh, make_results_table.py
tests/              unit tests for the non-TensorFlow logic
results/            figures, metrics.csv, runs.csv, preds, threshold.json, audit files
report/             REPORT.md skeleton and the final PDF
```

## Setup

Google Colab (recommended, GPU runtime):

1. Open any notebook from `notebooks/` in Colab. Set `REPO_URL` in the first cell, or upload the repository folder.
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets or environment variables for the dataset download.
3. With `USE_DRIVE = True` the notebooks keep weights and results on Google Drive so a disconnect does not lose work. Finished runs are skipped when a notebook is re-run.

Local:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export XRAY_DATA_ROOT=/path/to/chest_xray   # optional, otherwise kagglehub downloads it
python -m pytest -q tests
python scripts/smoke_test.py                # full pipeline on a tiny synthetic dataset
```

## How to run

Run the notebooks in order: `01` → `02` → `03` → `04`, or `bash scripts/run_all.sh` to execute all four and keep their outputs. Run `python scripts/smoke_test.py` first on a new machine to confirm the pipeline works.

Environment variables: `XRAY_DATA_ROOT`, `XRAY_RESULTS`, `XRAY_ARTIFACTS`, `XRAY_SPLIT_DIR`.

Rough GPU time on a Colab T4: notebook 01 a few minutes, notebook 02 about 1–2 h with 3 seeds × 6 strategies plus ablations, notebook 03 about 1–2 h, notebook 04 about 15 min. Reduce `SEEDS` or `EPOCHS` for a quicker first pass.

## Reproducibility

- Seeds: `src/config.py` (`SEED = 42`, `SEEDS = (42, 43, 44)`); `set_seed` seeds Python, NumPy and TensorFlow.
- Library versions are written to `results/env.txt` by notebook 01.
- Model weights are not committed. Weights are saved as `.weights.h5` under `artifacts/` (or Google Drive) and are rebuilt from code at load time.
- Every figure and number in the report comes from a file in `results/`.

## Contribution table

| Member | GitHub username | Responsibilities | Key files / experiments |
|---|---|---|---|
| Suraj | | | |
| Anurag (team leader) | | | |
| Vineet | | | |
| Meghana S | | | |

Each member commits from their own GitHub account on feature branches and opens pull requests.

## AI assistance

An AI assistant (Claude, Anthropic) was used to draft and refactor parts of the code, notebooks and documentation. List here exactly which parts, and confirm that every member can explain the code they own.

## Limitations

- Single public dataset from one pediatric center; labels come from the dataset curators, not from our own review.
- Group IDs are parsed from filenames and are a best-effort proxy for patients; the audit in `results/data_audit.json` reports how many images fall back to their own group and how many groups are shared with the official test set.
- The official test set has a lower pneumonia prevalence than the training data, so precision and predictive values shift; `results/screening_table.csv` shows outcomes at other prevalences.
- Grad-CAM maps are coarse and show where the model looked, not clinical reasoning.
- The test set was evaluated 5 times before this pipeline was built (recorded as legacy evaluations in `results/test_eval_log.json`).

## References

- Kermany et al., 2018, Cell: dataset source.
- Rajpurkar et al., 2017: CheXNet.
- Selvaraju et al., 2017: Grad-CAM.
- Zech et al., 2018, PLoS Medicine: shortcuts in pneumonia detection from chest radiographs.
- Lin et al., 2017: focal loss.
- Guo et al., 2017: calibration of modern neural networks.
