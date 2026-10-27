"""Step 1 of the pipeline: run the model once per question, grade it, and save everything.

Usage:
    python src/extract.py                 # 200 questions (Week 8)
    python src/extract.py --n 5000        # full run (Week 9)

Saves to data/<run name>/ (gitignored, the files are large):
    meta.csv       one row per question: id, subject, category, answer, pick, right/wrong, confidence...
    stage1.npy     hidden states after the question only          shape (N, layers+1, hidden)
    stage2.npy     hidden states after question + options        same shape
    stage3.npy     hidden states after the model's chosen letter  same shape
    run_info.json  model, seed, prompt version, library versions, timing

Row i of meta.csv matches row i of every stageX.npy.
Run this once. Experiments load the saved files and never rerun the model.
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os, sys, json, time, argparse, platform
# Use the already-downloaded model and dataset; don't check Hugging Face for updates.
# Must be set before importing datasets/transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import load_mmlu, sample_qids, category_of
from prompts import LETTERS, PROMPT_VERSION, stage1, stage2, stage3

# ---------------------------------------------------------------------------
# 2. Settings
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="number of questions")
parser.add_argument("--seed", type=int, default=0, help="random seed for picking questions")
parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--out", default=None, help="output folder (default: data/<model>_n<N>_seed<seed>)")
parser.add_argument("--force", action="store_true", help="redo the extraction even if the folder already has data")
args = parser.parse_args()

run_name = f"{args.model.split('/')[-1].lower()}_n{args.n}_seed{args.seed}"
out = args.out or os.path.join("data", run_name)
# meta.csv is written last, so if it exists the run finished. Don't overwrite it by accident.
if os.path.exists(os.path.join(out, "meta.csv")) and not args.force:
    sys.exit(f"{out}/ is already extracted. Use --force to redo it.")
os.makedirs(out, exist_ok=True)

if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16   # Apple GPU
else:
    device, dtype = "cpu", torch.bfloat16

# ---------------------------------------------------------------------------
# 3. Load the model and the questions
# ---------------------------------------------------------------------------
t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
n_layers, hidden = model.config.num_hidden_layers, model.config.hidden_size
print(f"model={args.model} layers={n_layers} hidden={hidden} device={device} load={time.time()-t0:.0f}s")

letter_ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
assert all(len(x) == 1 for x in letter_ids), letter_ids  # each letter must be exactly one token
letter_ids = [x[0] for x in letter_ids]

ds = load_mmlu()
qids = sample_qids(ds, args.n, args.seed)
print(f"{args.n} questions (seed {args.seed}) -> {out}")

# ---------------------------------------------------------------------------
# 4. Output files
# The stage files are written straight to disk as we go (open_memmap), so a
# 5,000-question run doesn't have to fit in memory. float16 halves the size.
# ---------------------------------------------------------------------------
shape = (args.n, n_layers + 1, hidden)  # +1: index 0 is the embedding layer
stages = {s: np.lib.format.open_memmap(os.path.join(out, f"stage{s}.npy"), mode="w+",
                                       dtype=np.float16, shape=shape) for s in (1, 2, 3)}

@torch.no_grad()
def run(text):
    """One forward pass. Returns the next-token logits and the last token's
    hidden state from every layer, shape (layers+1, hidden)."""
    ids = tok(text, return_tensors="pt").to(device)
    out = model(**ids, output_hidden_states=True)
    hs = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
    return out.logits[0, -1].float().cpu(), hs

# ---------------------------------------------------------------------------
# 5. Main loop: 3 passes per question
# ---------------------------------------------------------------------------
rows = []
t1 = time.time()
for row, qid in enumerate(qids):
    ex = ds[qid]

    _, stages[1][row] = run(stage1(tok, ex))           # stage 1: question only
    logits, stages[2][row] = run(stage2(tok, ex))      # stage 2: this pass also GRADES the model
    probs = torch.softmax(logits[letter_ids], dim=0)   # probabilities for A/B/C/D
    pick = int(probs.argmax())
    _, stages[3][row] = run(stage3(tok, ex, pick))     # stage 3: after the model's answer

    rows.append({
        "qid": qid,
        "subject": ex["subject"],
        "category": category_of(ex["subject"]),
        "gold": ex["answer"],                          # correct answer, 0-3 = A-D
        "pick": pick,                                  # model's answer, 0-3 = A-D
        "label": int(pick != ex["answer"]),            # 1 = wrong, 0 = right
        "confidence": float(probs.max()),              # the baseline to beat
        **{f"p{L}": float(p) for L, p in zip(LETTERS, probs)},
        "top_token_is_letter": int(logits.argmax()) in letter_ids,
        "question_len_tokens": len(tok.encode(ex["question"], add_special_tokens=False)),
    })

    done = row + 1
    if done % 25 == 0 or done == args.n:
        per = (time.time() - t1) / done
        print(f"  {done}/{args.n}  {per:.2f}s/question  ~{per*(args.n-done)/60:.1f} min left")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
for arr in stages.values():
    arr.flush()
meta = pd.DataFrame(rows)
meta.to_csv(os.path.join(out, "meta.csv"), index=False)

info = {
    "model": args.model, "n": args.n, "seed": args.seed, "prompt_version": PROMPT_VERSION,
    "device": device, "dtype": str(dtype), "layers": n_layers, "hidden": hidden,
    "seconds": round(time.time() - t1), "python": platform.python_version(),
    "torch": torch.__version__, "transformers": transformers.__version__,
}
with open(os.path.join(out, "run_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"\naccuracy {1 - meta.label.mean():.3f}  (wrong={meta.label.sum()}, right={(meta.label == 0).sum()})")
print(f"model's top token was one of A/B/C/D: {meta.top_token_is_letter.mean():.1%}")
print(f"saved to {out}/  ({sum(a.nbytes for a in stages.values())/1e6:.0f} MB)")
print(f"next: python src/make_splits.py {out}")
