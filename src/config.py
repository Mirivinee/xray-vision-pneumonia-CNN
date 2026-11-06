import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = Path(os.environ.get("XRAY_SPLIT_DIR", DATA_DIR / "splits"))
MANIFEST_PATH = SPLIT_DIR / "manifest.csv"

RESULTS_DIR = Path(os.environ.get("XRAY_RESULTS", PROJECT_ROOT / "results"))
FIG_DIR = RESULTS_DIR / "figures"
PRED_DIR = RESULTS_DIR / "preds"
HIST_DIR = RESULTS_DIR / "history"
RUNS_CSV = RESULTS_DIR / "runs.csv"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
THRESHOLD_JSON = RESULTS_DIR / "threshold.json"
AUDIT_JSON = RESULTS_DIR / "data_audit.json"
TEST_LOG_JSON = RESULTS_DIR / "test_eval_log.json"
IMAGE_STATS_CSV = RESULTS_DIR / "image_stats.csv"

ARTIFACT_DIR = Path(os.environ.get("XRAY_ARTIFACTS", PROJECT_ROOT / "artifacts"))
WEIGHT_DIR = ARTIFACT_DIR / "weights"
THUMB_NPZ = ARTIFACT_DIR / "thumbs.npz"

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SEED = 42
SEEDS = (42, 43, 44)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
N_VAL_FOLDS = 10

TARGET_SENSITIVITY = 0.97
SENSITIVITY_TARGETS = (0.90, 0.95, 0.97, 0.99)
PREVALENCES = (0.05, 0.20, 0.626)
LEGACY_TEST_EVALUATIONS = 5


def ensure_dirs():
    for d in (SPLIT_DIR, RESULTS_DIR, FIG_DIR, PRED_DIR, HIST_DIR, ARTIFACT_DIR, WEIGHT_DIR):
        d.mkdir(parents=True, exist_ok=True)
