"""Week 1: run Qwen on one MMLU question and print its A/B/C/D probabilities.

Usage:
    python scripts/one_question.py          # random question
    python scripts/one_question.py 1234     # question #1234 of the MMLU test set
"""
import os, sys, random
# Use the already-downloaded model and dataset; don't check Hugging Face for updates.
# Must be set before importing datasets/transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LETTERS = ["A", "B", "C", "D"]

if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.bfloat16

print(f"loading {MODEL} on {device}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

ds = load_dataset("cais/mmlu", "all", split="test")
i = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(len(ds))
ex = ds[i]

# The fixed project prompt (DAIS_PROJECT_OVERVIEW.md, Section 4)
opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, ex["choices"]))
user = ("Answer the following multiple choice question. Reply with only the letter (A, B, C, or D).\n\n"
        f"Question: {ex['question']}\n{opts}")
text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

with torch.no_grad():
    logits = model(**tok(text, return_tensors="pt").to(device)).logits[0, -1].float().cpu()
probs = torch.softmax(logits[letter_ids], dim=0)
pick, gold = int(probs.argmax()), ex["answer"]

print(f"\nMMLU #{i} ({ex['subject']})\n")
print(f"Question: {ex['question']}\n{opts}\n")
for L, p in zip(LETTERS, probs):
    print(f"  {L}: {p:.1%}  {'#' * int(p * 40)}")
print(f"\nModel picked {LETTERS[pick]} ({probs[pick]:.1%} confident)")
print(f"Correct answer: {LETTERS[gold]}  ->  {'RIGHT' if pick == gold else 'WRONG'}")
