import json
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def set_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not serializable: {type(o)}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_env(path=None):
    path = Path(path) if path else C.RESULTS_DIR / "env.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
    for name in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "PIL", "tensorflow", "keras", "kagglehub"):
        try:
            mod = __import__(name)
            lines.append(f"{name} {getattr(mod, '__version__', 'unknown')}")
        except Exception:
            lines.append(f"{name} not installed")
    try:
        import tensorflow as tf

        lines.append(f"gpus {[g.name for g in tf.config.list_physical_devices('GPU')]}")
    except Exception:
        pass
    path.write_text("\n".join(lines) + "\n")
    return lines


def sync_results_to_repo():
    src = C.RESULTS_DIR.resolve()
    dst = (C.PROJECT_ROOT / "results").resolve()
    if src != dst:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def run_id(*parts):
    return "-".join(str(p) for p in parts if p not in (None, ""))


def weights_path(rid):
    return C.WEIGHT_DIR / f"{rid}.weights.h5"


def load_runs():
    if C.RUNS_CSV.exists():
        return pd.read_csv(C.RUNS_CSV, keep_default_na=False, na_values=[""])
    return pd.DataFrame()


def log_run(row):
    df = load_runs()
    new = pd.DataFrame([row])
    if len(df):
        df = df[df["run_id"] != row["run_id"]]
        df = pd.concat([df, new], ignore_index=True) if len(df) else new
    else:
        df = new
    C.RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RUNS_CSV, index=False)


def get_run(rid):
    df = load_runs()
    if df.empty or rid not in set(df["run_id"]):
        return None
    return df[df["run_id"] == rid].iloc[0].to_dict()


def is_done(rid):
    return get_run(rid) is not None and weights_path(rid).exists()
