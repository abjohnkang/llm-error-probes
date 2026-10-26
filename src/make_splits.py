"""Step 2 of the pipeline: make the fixed train/val/test split.

Usage:
    python src/make_splits.py data/qwen2.5-3b-instruct_n200_seed0

Writes splits.json into the same folder: 70% train, 10% val, 20% test,
listed by question id (qid). The whole team uses this one file:
  - train: the probe learns from these
  - val:   for choosing settings (like C) later; never use test for that
  - test:  scored only at the end, for the numbers we report

The split is stratified by right/wrong and by category, so each part has the
same mix of mistakes and subjects.
"""
import os, sys, json
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 0
folder = sys.argv[1]
meta = pd.read_csv(os.path.join(folder, "meta.csv"))

def split(strata):
    train, rest = train_test_split(meta.qid, test_size=0.3, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=2/3, stratify=strata[rest.index], random_state=SEED)
    return train, val, test

# Stratify on label + category. With few questions some groups are too small
# to split, so fall back to label only.
try:
    train, val, test = split(meta.label.astype(str) + "_" + meta.category)
except ValueError:
    print("some label x category groups are too small -> stratifying by label only")
    train, val, test = split(meta.label)

splits = {"seed": SEED, "train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())}
with open(os.path.join(folder, "splits.json"), "w") as f:
    json.dump(splits, f)

for name in ("train", "val", "test"):
    part = meta[meta.qid.isin(splits[name])]
    print(f"{name:5s} {len(part):5d} questions  {part.label.mean():.0%} wrong")
print(f"next: python src/probe.py {folder}")
