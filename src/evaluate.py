"""Scoring: AUROC with a 95% bootstrap confidence interval.

AUROC: how well a score ranks wrong answers above right ones.
0.5 = coin flip, 1.0 = perfect.

The confidence interval shows how much the AUROC could move just by chance,
given how many test questions we have. Fewer test questions -> wider interval.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_ci(y, score, n_boot=1000, seed=0):
    """Return (auroc, low, high). Higher score should mean "more likely wrong".

    Bootstrap: resample the test questions with replacement 1,000 times,
    recompute AUROC each time, and take the middle 95% of those values.
    """
    y, score = np.asarray(y), np.asarray(score)
    auc = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():  # resample has only right or only wrong answers -> skip
            continue
        boots.append(roc_auc_score(y[i], score[i]))
    low, high = np.percentile(boots, [2.5, 97.5])
    return auc, low, high
