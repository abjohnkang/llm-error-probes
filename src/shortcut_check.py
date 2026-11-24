"""Does the probe add anything beyond confidence, subject and question + options wording?

Usage:
    python src/shortcut_check.py data/qwen2.5-3b-instruct_n14042_seed0

The probe beats the model's own confidence. But maybe it only learned shortcuts
that are easy to get without looking inside the model: which subject the
question is from, or what the question and its options say. To test this, we
first combine confidence with those shortcuts, then check whether adding the
probe on top still helps.

Saves results/<run name>/shortcut_check.csv
"""
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import auroc_ci, paired_diff_ci
from data import load_mmlu

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

# One row per detector for the results CSV (section 6). Confidence is compared
# to nothing, so its difference columns stay empty.
rows = [{"detector": "conf", "auroc": auc, "ci_low": lo, "ci_high": hi}]

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

# Text: the question AND its four options (baselines.py uses the question only).
# meta.csv has no text, so look it up in MMLU by qid (the same dataset and row
# numbers extract.py used). Same TF-IDF settings as baselines.py.
# Watch the two train numbers: the model scores the questions it was fitted on
# far too well, because it memorized them. Those are the scores the combiner
# must never learn from.
ds = load_mmlu()
texts = np.array([ds[int(q)]["question"] + " " + " ".join(ds[int(q)]["choices"])
                  for q in meta.qid], dtype=object)
text_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000),
)
text_train, text_test = cv_scores("text", text_model, texts)

# ---------------------------------------------------------------------------
# 4. Combine confidence + subject + text, and compare to confidence alone
# Same combiner as in probe.py: standardize, then logistic regression.
# Subject and text go in as log-odds (conf is already a log), so all inputs
# are on a log scale, and standardizing keeps any one from being ignored just
# because its values are spread over a smaller range.
# ---------------------------------------------------------------------------
def logit(p):
    """Probability -> log-odds. Spreads out values near 0 and 1.
    (Same as in probe.py. Not imported from there: probe.py is a script, so
    importing it would run it.)"""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def combine(name, train_inputs, test_inputs):
    """Fit the combiner on train, print its test AUROC, and return
    (test scores, AUROC, CI low, CI high).
    Every train input must come from cv_scores (cross-validated)."""
    combiner = make_pipeline(StandardScaler(), LogisticRegression())
    combiner.fit(np.column_stack(train_inputs), y[train])
    scores = combiner.predict_proba(np.column_stack(test_inputs))[:, 1]
    auc, lo, hi = auroc_ci(y[test], scores)
    print(f"{name:30s} test AUROC {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
    # Self-check: the combiner gets confidence as one input, so it should not
    # do worse than confidence alone. If it does, an input was probably scored
    # on its own train questions instead of cross-validated.
    if auc < roc_auc_score(y[test], conf[test]):
        print("  WARNING: below confidence alone. Is every train input cross-validated?")
    return scores, auc, lo, hi

def compare(name_a, a, name_b, b):
    """Paired bootstrap on the same test questions: does a beat b?
    Returns the difference columns for the results CSV."""
    d, lo, hi, p = paired_diff_ci(y[test], a, b)
    print(f"  {name_a} - {name_b}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    return {"compared_to": name_b, "diff": d, "diff_low": lo, "diff_high": hi, "p": p}

print()
shortcuts, auc, lo, hi = combine("conf + subject + text",
                                 [conf[train], logit(subject_train), logit(text_train)],
                                 [conf[test], logit(subject_test), logit(text_test)])
diff = compare("conf + subject + text", shortcuts, "conf", conf[test])
rows.append({"detector": "conf + subject + text", "auroc": auc, "ci_low": lo, "ci_high": hi, **diff})

# ---------------------------------------------------------------------------
# 5. Add the probe on top: does it still help?
# For each stage, use the layer probe.py chose on val (read from probe_best.csv,
# so no layer number is typed in by hand). Same probe as probe.py (C=0.01).
# Its train scores are cross-validated too, like every other combiner input.
# If conf + subject + text + probe beats conf + subject + text, the hidden
# states hold error information that subject and wording don't explain.
# ---------------------------------------------------------------------------
run = os.path.basename(folder)
best = pd.read_csv(os.path.join("results", run, "probe_best.csv"))
for _, row in best.iterrows():
    stage, layer = int(row.stage), int(row.layer)
    print(f"\nstage {stage}, layer {layer}")
    X = np.load(os.path.join(folder, f"stage{stage}.npy"), mmap_mode="r")
    Xl = np.asarray(X[:, layer], dtype=np.float32)  # one layer: (questions, 2048)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    probe_train, probe_test = cv_scores("probe", probe, Xl)
    with_probe, auc, lo, hi = combine("conf + subject + text + probe",
                                      [conf[train], logit(subject_train), logit(text_train), logit(probe_train)],
                                      [conf[test], logit(subject_test), logit(text_test), logit(probe_test)])
    diff = compare("with probe", with_probe, "conf + subject + text", shortcuts)
    rows.append({"detector": "conf + subject + text + probe", "stage": stage, "layer": layer,
                 "auroc": auc, "ci_low": lo, "ci_high": hi, **diff})

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out = os.path.join("results", run)
os.makedirs(out, exist_ok=True)
# Int64 keeps stage and layer as whole numbers, with empty cells where a row has none.
df = pd.DataFrame(rows).astype({"stage": "Int64", "layer": "Int64"})
df = df[["detector", "stage", "layer", "auroc", "ci_low", "ci_high",
         "compared_to", "diff", "diff_low", "diff_high", "p"]]
df.to_csv(os.path.join(out, "shortcut_check.csv"), index=False)
print(f"\nsaved {out}/shortcut_check.csv")
