"""Step 3 of the pipeline: train a probe on every layer and score it on the test set.

Usage:
    python src/probe.py data/qwen2.5-3b-instruct_n200_seed0

For each stage and each layer: train logistic regression on the train split,
predict "probability this answer is wrong" for the test split, and score it
with AUROC (+ 95% confidence interval). The model's own confidence is scored
on the same test questions as the baseline to beat.

Saves results/<run name>/probe_layers.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import auroc_ci

# ---------------------------------------------------------------------------
# 1. Load the saved data (no model needed)
# ---------------------------------------------------------------------------
folder = sys.argv[1].rstrip("/")
meta = pd.read_csv(os.path.join(folder, "meta.csv"))
with open(os.path.join(folder, "splits.json")) as f:
    splits = json.load(f)
train = meta.qid.isin(splits["train"]).values  # True/False per row
test = meta.qid.isin(splits["test"]).values
y = meta.label.values                          # 1 = wrong, 0 = right
print(f"train {train.sum()}  test {test.sum()}")

# ---------------------------------------------------------------------------
# 2. Baseline: the model's own confidence (minus sign: low confidence -> likely wrong)
# ---------------------------------------------------------------------------
base, base_lo, base_hi = auroc_ci(y[test], -meta.confidence.values[test])
print(f"BASELINE confidence AUROC {base:.3f}  [{base_lo:.2f}, {base_hi:.2f}]\n")

# ---------------------------------------------------------------------------
# 3. Probe every stage x layer
# C=0.01 = strong regularization, same as the pilot. Choosing C on the val
# split comes later.
# ---------------------------------------------------------------------------
results = []
for stage, name in [(1, "question only"), (2, "question+options"), (3, "after answer")]:
    X = np.load(os.path.join(folder, f"stage{stage}.npy"), mmap_mode="r")
    for layer in range(X.shape[1]):
        Xl = np.asarray(X[:, layer], dtype=np.float32)
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        probe.fit(Xl[train], y[train])
        p_wrong = probe.predict_proba(Xl[test])[:, 1]
        auc, lo, hi = auroc_ci(y[test], p_wrong)
        results.append({"stage": stage, "layer": layer, "auroc": auc, "ci_low": lo, "ci_high": hi})

    # Print every 4th layer plus the best one
    df = pd.DataFrame([r for r in results if r["stage"] == stage])
    best = df.loc[df.auroc.idxmax()]
    row = "  ".join(f"L{r.layer}:{r.auroc:.2f}" for r in df.itertuples() if r.layer % 4 == 0)
    print(f"stage {stage} ({name:16s}) {row}")
    print(f"  best: layer {int(best.layer)}  AUROC {best.auroc:.3f}  [{best.ci_low:.2f}, {best.ci_high:.2f}]")

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
out = os.path.join("results", os.path.basename(folder))
os.makedirs(out, exist_ok=True)
df = pd.DataFrame(results)
df.insert(0, "baseline_confidence", base)
df.to_csv(os.path.join(out, "probe_layers.csv"), index=False)
print(f"\nsaved {out}/probe_layers.csv")
print("Note: with few test questions the intervals are wide. Don't pick a 'best layer' from a small run.")
