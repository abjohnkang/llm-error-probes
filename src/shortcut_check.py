"""Does the probe add anything beyond confidence, subject and question + options wording?

Usage:
    python src/shortcut_check.py data/qwen2.5-3b-instruct_n14042_seed0

The probe beats the model's own confidence. But maybe it only learned shortcuts
that are easy to get without looking inside the model: which subject the
question is from, or what the question and its options say. To test this, we
first combine confidence with those shortcuts, then check whether adding the
probe on top still helps.
"""
import os, sys, json
import numpy as np
import pandas as pd

from evaluate import auroc_ci

# ---------------------------------------------------------------------------
# 1. Load the saved data (no model needed)
# ---------------------------------------------------------------------------
folder = sys.argv[1].rstrip("/")
meta = pd.read_csv(os.path.join(folder, "meta.csv"))
with open(os.path.join(folder, "splits.json")) as f:
    splits = json.load(f)
train = meta.qid.isin(splits["train"]).values  # True/False per row
val = meta.qid.isin(splits["val"]).values
test = meta.qid.isin(splits["test"]).values
y = meta.label.values                          # 1 = wrong, 0 = right
print(f"train {train.sum()}  val {val.sum()}  test {test.sum()}")

# ---------------------------------------------------------------------------
# 2. Baseline: the model's own confidence (same score as probe.py)
# About a third of answers have confidence saved as exactly 1.0 (rounded), so
# we use log(sum of the other 3 letters' probabilities), computed precisely.
# Higher = less confident -> more likely wrong.
# ---------------------------------------------------------------------------
probs = np.sort(meta[["pA", "pB", "pC", "pD"]].values, axis=1)
conf = np.log(probs[:, :3].sum(axis=1))
auc, lo, hi = auroc_ci(y[test], conf[test])
print(f"confidence   test AUROC {auc:.3f}  [{lo:.2f}, {hi:.2f}]")
