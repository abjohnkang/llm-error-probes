import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
test = meta.qid.isin(splits["test"]).values    # True/False per row
y = meta.label.values                          # 1 = wrong, 0 = right
print(f"train {train.sum()}  test {test.sum()}\n")

results = []

def add_result(name, score):
    """Score one baseline on the test split and save the result."""
    auc, lo, hi = auroc_ci(y[test], score[test])
    results.append({"baseline": name, "auroc": auc, "ci_low": lo, "ci_high": hi})
    print(f"{name:12s} AUROC {auc:.3f}  [{lo:.2f}, {hi:.2f}]")

# ---------------------------------------------------------------------------
# 2. Baselines that need no training
# ---------------------------------------------------------------------------
probs = meta[["pA", "pB", "pC", "pD"]].values
sorted_probs = np.sort(probs, axis=1)
margin = sorted_probs[:, -1] - sorted_probs[:, -2]
entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)

add_result("confidence", -meta.confidence.values)  # should match probe.py
add_result("margin", -margin)                      # small gap -> likely wrong
add_result("entropy", entropy)                     # spread out -> likely wrong
add_result("length", meta.question_len_tokens.values.astype(float))  # longer -> likely wrong

# ---------------------------------------------------------------------------
# 3. Simple trained baselines (fit on train only)
# ---------------------------------------------------------------------------
def add_categorical_baseline(column):
    """Predict wrong/right from one categorical column in meta.csv."""
    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=2000),
    )
    X = meta[[column]]
    model.fit(X[train], y[train])
    p_wrong = model.predict_proba(X)[:, 1]
    add_result(column, p_wrong)

add_categorical_baseline("pick")
add_categorical_baseline("category")

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
out = os.path.join("results", os.path.basename(folder))
os.makedirs(out, exist_ok=True)
pd.DataFrame(results).to_csv(os.path.join(out, "baselines.csv"), index=False)
print(f"\nsaved {out}/baselines.csv")