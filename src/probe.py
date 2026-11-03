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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import auroc_ci, paired_diff_ci

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
# 2. Baseline: the model's own confidence (minus sign: low confidence -> likely wrong)
# ---------------------------------------------------------------------------
conf = -meta.confidence.values
base, base_lo, base_hi = auroc_ci(y[test], conf[test])

def logit(p):
    """Probability -> log-odds. Spreads out values near 0 and 1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
print(f"BASELINE confidence AUROC {base:.3f}  [{base_lo:.2f}, {base_hi:.2f}]\n")

# ---------------------------------------------------------------------------
# 3. Probe every stage x layer
# C=0.01 = strong regularization, same as the pilot. Choosing C on the val
# split comes later.
# ---------------------------------------------------------------------------
results, bests = [], []
for stage, name in [(1, "question only"), (2, "question+options"), (3, "after answer")]:
    X = np.load(os.path.join(folder, f"stage{stage}.npy"), mmap_mode="r")
    test_preds = {}  # layer -> the probe's test predictions, kept for the comparison below
    for layer in range(X.shape[1]):
        Xl = np.asarray(X[:, layer], dtype=np.float32)
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        probe.fit(Xl[train], y[train])
        # Val score: used later to choose the best layer without looking at test
        val_auc = roc_auc_score(y[val], probe.predict_proba(Xl[val])[:, 1])
        p_wrong = probe.predict_proba(Xl[test])[:, 1]
        test_preds[layer] = p_wrong
        auc, lo, hi = auroc_ci(y[test], p_wrong)
        results.append({"stage": stage, "layer": layer, "val_auroc": val_auc,
                        "auroc": auc, "ci_low": lo, "ci_high": hi})

    # Print every 4th layer plus the best one.
    # The best layer is chosen by its VAL score, then reported with its TEST score.
    # Picking the top test score out of 37 layers would be a little lucky (too high).
    df = pd.DataFrame([r for r in results if r["stage"] == stage])
    best = df.loc[df.val_auroc.idxmax()]
    row = "  ".join(f"L{r.layer}:{r.auroc:.2f}" for r in df.itertuples() if r.layer % 4 == 0)
    print(f"stage {stage} ({name:16s}) {row}")
    print(f"  best (chosen on val): layer {int(best.layer)}  test AUROC {best.auroc:.3f}  [{best.ci_low:.2f}, {best.ci_high:.2f}]")

    # Does the chosen layer beat confidence? Paired bootstrap on the same test questions.
    # If the whole interval is above 0, the probe wins.
    diff, d_lo, d_hi, p = paired_diff_ci(y[test], test_preds[int(best.layer)], conf[test])
    print(f"  probe - confidence: {diff:+.3f}  [{d_lo:+.3f}, {d_hi:+.3f}]  p={p:.3f}")

    # Combined detector ("stacking"): a tiny second model that combines two numbers,
    # the probe's prediction and confidence, and learns how much to trust each.
    # If it beats confidence alone, the hidden states add something confidence doesn't have.
    # It learns from the probe's predictions on TRAIN questions, made with 5-fold
    # cross-validation so each one comes from a probe that never saw that question.
    # (The probe's predictions on its own training questions are too good, and the
    # combiner would trust them too much.)
    # Both numbers go in as log-odds, then get standardized, so neither one is
    # ignored just because its values are spread over a smaller range.
    Xl = np.asarray(X[:, int(best.layer)], dtype=np.float32)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    train_preds = cross_val_predict(probe, Xl[train], y[train], cv=5, method="predict_proba")[:, 1]
    conf_lo = logit(meta.confidence.values)
    combo = make_pipeline(StandardScaler(), LogisticRegression())
    combo.fit(np.column_stack([logit(train_preds), conf_lo[train]]), y[train])
    combo_preds = combo.predict_proba(np.column_stack([logit(test_preds[int(best.layer)]), conf_lo[test]]))[:, 1]
    c_auc, c_lo, c_hi = auroc_ci(y[test], combo_preds)
    print(f"  combined (layer + confidence): test AUROC {c_auc:.3f}  [{c_lo:.2f}, {c_hi:.2f}]")

    # Does the combination beat each part on its own? Same paired bootstrap as above.
    # Beats confidence -> the hidden states add something confidence doesn't have.
    # Beats the probe  -> confidence adds something the probe doesn't have.
    cc, cc_lo, cc_hi, cc_p = paired_diff_ci(y[test], combo_preds, conf[test])
    cp, cp_lo, cp_hi, cp_p = paired_diff_ci(y[test], combo_preds, test_preds[int(best.layer)])
    print(f"  combined - confidence: {cc:+.3f}  [{cc_lo:+.3f}, {cc_hi:+.3f}]  p={cc_p:.3f}")
    print(f"  combined - probe:      {cp:+.3f}  [{cp_lo:+.3f}, {cp_hi:+.3f}]  p={cp_p:.3f}")

    bests.append({**best.to_dict(), "stage": stage, "layer": int(best.layer), "diff_vs_conf": diff, "diff_low": d_lo, "diff_high": d_hi, "p": p,
                  "combo_auroc": c_auc, "combo_ci_low": c_lo, "combo_ci_high": c_hi,
                  "combo_vs_conf": cc, "combo_vs_conf_low": cc_lo, "combo_vs_conf_high": cc_hi, "combo_vs_conf_p": cc_p,
                  "combo_vs_probe": cp, "combo_vs_probe_low": cp_lo, "combo_vs_probe_high": cp_hi, "combo_vs_probe_p": cp_p})

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
out = os.path.join("results", os.path.basename(folder))
os.makedirs(out, exist_ok=True)
df = pd.DataFrame(results)
df.insert(0, "baseline_confidence", base)
df.to_csv(os.path.join(out, "probe_layers.csv"), index=False)
pd.DataFrame(bests).to_csv(os.path.join(out, "probe_best.csv"), index=False)  # one row per stage
print(f"\nsaved {out}/probe_layers.csv and probe_best.csv")
print("Note: with few val/test questions the chosen layer and intervals are unreliable. Read them on the full run.")
