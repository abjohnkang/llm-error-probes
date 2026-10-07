"""Pilot: does the 'Do LLMs know when they're wrong?' MCQ pipeline actually run?

Grade MMLU questions by comparing the model's A/B/C/D scores to the answer key,
save last-token hidden states from every layer at 3 stages, then train a
logistic-regression probe per layer and compare it to the model's own confidence.
"""
import sys, time, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
LETTERS = ["A", "B", "C", "D"]
device = "mps" if torch.backends.mps.is_available() else "cpu"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
n_layers = model.config.num_hidden_layers
print(f"model={MODEL} layers={n_layers} hidden={model.config.hidden_size} device={device} load={time.time()-t0:.0f}s")

letter_ids = [tok.encode(l, add_special_tokens=False) for l in LETTERS]
assert all(len(x) == 1 for x in letter_ids), letter_ids
letter_ids = [x[0] for x in letter_ids]

ds = load_dataset("cais/mmlu", "all", split="test")
idx = random.Random(0).sample(range(len(ds)), N)

def chat(user, assistant_prefix=""):
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + assistant_prefix

@torch.no_grad()
def run(text):
    ids = tok(text, return_tensors="pt").to(device)
    out = model(**ids, output_hidden_states=True)
    # hidden_states: embeddings + one per layer -> take last token of each
    hs = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
    return out.logits[0, -1].float().cpu(), hs

labels, conf, top_is_letter, subjects = [], [], [], []
feats = {1: [], 2: [], 3: []}
t1 = time.time()
for k, i in enumerate(idx):
    ex = ds[i]
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, ex["choices"]))
    instr = "Answer the following multiple choice question. Reply with only the letter (A, B, C, or D).\n\n"

    # Stage 1: question only (no options)
    _, hs1 = run(chat(f"Question: {ex['question']}"))
    # Stage 2: question + options -> this pass also GRADES the model
    logits, hs2 = run(chat(instr + f"Question: {ex['question']}\n{opts}"))
    probs = torch.softmax(logits[letter_ids], dim=0)
    pick = int(probs.argmax())
    # Stage 3: after the model commits to its letter
    _, hs3 = run(chat(instr + f"Question: {ex['question']}\n{opts}", LETTERS[pick]))

    labels.append(int(pick != ex["answer"]))  # 1 = wrong, 0 = right
    conf.append(float(probs.max()))
    top_is_letter.append(int(logits.argmax()) in letter_ids)
    subjects.append(ex["subject"])
    feats[1].append(hs1); feats[2].append(hs2); feats[3].append(hs3)
    if k == 9:
        per = (time.time() - t1) / 10
        print(f"~{per:.2f}s per question (3 passes) -> est. {per*N/60:.1f} min for N={N}")

y = np.array(labels)
print(f"\nextraction time {time.time()-t1:.0f}s")
print(f"accuracy {1-y.mean():.3f}  (wrong={y.sum()}, right={len(y)-y.sum()})")
print(f"model's top token was one of A/B/C/D: {np.mean(top_is_letter):.1%}")
print(f"feature size per stage: {np.array(feats[2]).nbytes/1e6:.1f} MB for {N} questions")

# Baseline: the model's own confidence (low confidence -> predict wrong)
print(f"\nBASELINE output-confidence AUROC: {roc_auc_score(y, -np.array(conf)):.3f}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
def probe_auc(X):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

layers = sorted(set(np.linspace(0, n_layers, 9).astype(int)))
for stage, name in [(1, "question only"), (2, "question+options"), (3, "after answer")]:
    X = np.array(feats[stage])
    row = "  ".join(f"L{l}:{probe_auc(X[:, l]):.2f}" for l in layers)
    print(f"stage {stage} ({name:16s}) {row}")
print(f"\ntotal {time.time()-t0:.0f}s")
