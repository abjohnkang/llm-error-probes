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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
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

# ---------------------------------------------------------------------------
# 3. Scores for the combiner, made the safe way
# Later, a small combiner model learns from TRAIN questions how much to trust
# each score. So each train question must be scored by a model that never saw
# it, or the combiner sees scores that are too good and trusts them too much.
# Rule: every input to the combiner gets its train scores from
# cross_val_predict (5 folds: each fifth of train is scored by a model fitted
# on the other four fifths). Test questions are scored by a model fitted on
# all of train.
# ---------------------------------------------------------------------------
def cv_scores(name, model, X):
    """Return (train scores, test scores) for one model, made the safe way.
    Also prints train AUROC both ways, to show why the safe way matters."""
    train_scores = cross_val_predict(model, X[train], y[train], cv=5,
                                     method="predict_proba")[:, 1]
    model.fit(X[train], y[train])
    seen_scores = model.predict_proba(X[train])[:, 1]  # the unsafe way: only printed
    test_scores = model.predict_proba(X[test])[:, 1]
    print(f"{name:12s} train AUROC: cross-validated {roc_auc_score(y[train], train_scores):.3f}, "
          f"same questions {roc_auc_score(y[train], seen_scores):.3f}   "
          f"test AUROC {roc_auc_score(y[test], test_scores):.3f}")
    return train_scores, test_scores

# Subject: same model as in baselines.py (one-hot subject -> logistic regression).
# handle_unknown="ignore": a subject never seen in train does not crash later.
subject_model = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                              LogisticRegression(max_iter=2000))
subject_train, subject_test = cv_scores("subject", subject_model,
                                        meta[["subject"]].astype(str).values)
