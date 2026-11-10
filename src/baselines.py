import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

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
print(f"train {train.sum()}  val {val.sum()}  test {test.sum()}\n")

results = []

def add_result(name, score):
    """Score one baseline on val and test, and save the result."""
    val_auc = roc_auc_score(y[val], score[val])
    auc, lo, hi = auroc_ci(y[test], score[test])
    results.append({"baseline": name, "val_auroc": val_auc,
                    "auroc": auc, "ci_low": lo, "ci_high": hi})
    print(f"{name:12s} val {val_auc:.3f}   test AUROC {auc:.3f}  [{lo:.2f}, {hi:.2f}]")

# ---------------------------------------------------------------------------
# 2. Baselines that need no training
# About a third of top probabilities are saved as exactly 1.0, so (like
# probe.py) the top probability is rebuilt as 1 - (sum of the other three).
# ---------------------------------------------------------------------------
probs = np.sort(meta[["pA", "pB", "pC", "pD"]].values, axis=1)  # smallest -> largest
rest = probs[:, :3].sum(axis=1)  # = 1 - top, computed precisely
top = 1 - rest
second = probs[:, 2]

conf = np.log(rest)                                   # same score as probe.py
margin = top - second
p_all = np.column_stack([probs[:, :3], top])
entropy = -(p_all * np.log(np.clip(p_all, 1e-12, None))).sum(axis=1)

add_result("confidence", conf)      # sanity check: should match probe.py's baseline
add_result("margin", -margin)       # small gap -> likely wrong
add_result("entropy", entropy)      # spread out -> likely wrong
add_result("length", meta.question_len_tokens.values.astype(float))  # longer -> likely wrong

# ---------------------------------------------------------------------------
# 3. Simple trained baselines: pick, category, subject (fit on train only)
# handle_unknown="ignore": a value never seen in train does not crash later.
# astype(str): pick is 0-3, and should be treated as a category, not a number.
# ---------------------------------------------------------------------------
def add_categorical_baseline(column):
    """Predict wrong/right from one categorical column in meta.csv."""
    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=2000),
    )
    X = meta[[column]].astype(str)
    model.fit(X[train], y[train])
    add_result(column, model.predict_proba(X)[:, 1])

add_categorical_baseline("pick")
add_categorical_baseline("category")

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
out = os.path.join("results", os.path.basename(folder))
os.makedirs(out, exist_ok=True)
pd.DataFrame(results).to_csv(os.path.join(out, "baselines.csv"), index=False)
print(f"\nsaved {out}/baselines.csv")