#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for nb in notebooks/01_eda.ipynb notebooks/02_baseline_cnn.ipynb notebooks/03_transfer_learning.ipynb notebooks/04_evaluation_and_explainability.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
done
python scripts/make_results_table.py
