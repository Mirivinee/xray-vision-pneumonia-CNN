import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402


def fmt(df, cols):
    out = df[cols].copy()
    for c in cols:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: f"{v:.3f}")
    header = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    rows = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in out.itertuples(index=False))
    return header + rows


def main():
    m = pd.read_csv(C.METRICS_CSV)
    cols = ["run_id", "fn", "fp", "accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]
    parts = []
    for setting, title in (("validation-chosen threshold", "Test set at the validation-chosen threshold"), ("0.5", "Test set at threshold 0.5")):
        t = m[(m["split"] == "test") & (m["setting"] == setting)]
        if len(t):
            parts.append(f"### {title}\n\n" + fmt(t.sort_values("roc_auc", ascending=False), cols))
    v = m[(m["split"] == "val")].sort_values("roc_auc", ascending=False).head(15)
    parts.append("### Validation (top 15 runs by ROC-AUC, threshold 0.5)\n\n" + fmt(v, cols))
    out = C.RESULTS_DIR / "summary.md"
    out.write_text("\n".join(parts))
    print(out)


if __name__ == "__main__":
    main()
