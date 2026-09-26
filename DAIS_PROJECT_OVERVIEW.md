# Do AI Models Know When They're Wrong?
### DAIS Club Research Project — Overview for Members

> **Status:** Planning is done. Next step: set up the repo and start Week 1.
>
> Short on time? Read the [TL;DR](#1-tldr).

---

## 1. TL;DR

- **Question:** When an AI model gives a wrong answer, does its *internal activity* reveal that it's likely wrong, even when it sounds confident?
- **How:** Give a free, open-source model (**Qwen2.5-3B-Instruct**) thousands of multiple-choice questions from **MMLU**, grade it automatically, record its internal "thoughts" (**hidden states**) at every layer, and train small classifiers (**probes**) that predict *"this answer will be wrong."*
- **We never train or modify the AI model.** It is frozen. The only ML we train is the small probe.
- **Main metric:** AUROC (0.5 = coin flip, 1.0 = perfect).
- **Baseline to beat:** the model's own confidence in its answer.
- **Duration:** one semester, optional second. **Team:** 3–4 people.

---

## 2. Research Questions

| # | Question | Plain version |
|---|---|---|
| RQ1 | Which layers carry the most information about whether the answer will be wrong? | Which "floor" of the AI's brain knows the most? |
| RQ2 | Can a hidden-state probe predict mistakes better than the model's own confidence? | Does the AI know more than it admits? |
| RQ3 | How early is a mistake detectable: after the question, after the options, or after answering? | Can we tell before it even tries? |
| RQ4 | Does a probe trained on some subjects (e.g., STEM) work on others (e.g., Humanities)? | Does the detector generalize? |

---

## 3. Key Concepts

| Term | Meaning here |
|---|---|
| **Layer** | The model processes text through a stack of layers (36 for Qwen2.5-3B), like floors in a building. |
| **Hidden state** | The model's internal representation at one layer and token: 2,048 numbers for Qwen2.5-3B. |
| **Frozen model** | We only run the model; its weights never change. |
| **Probe / detector** | A small classifier (e.g., logistic regression) trained on hidden states to predict right vs. wrong. |
| **Label** | 0 = model answered correctly, 1 = model answered incorrectly. |
| **Confidence** | Softmax probability of the letter the model picked (among A–D). Our main baseline. |
| **AUROC** | How well a detector ranks wrong answers above right ones. 0.5 = random, 1.0 = perfect. |
| **Shortcut** | A detector succeeding for a trivial reason (e.g., which letter was picked) instead of what we care about. |

---

## 4. Method

**Model:** `Qwen/Qwen2.5-3B-Instruct` from Hugging Face (36 layers, hidden size 2048, ~6 GB). Run in eval mode only.

**Dataset:** MMLU (`cais/mmlu`, config `all`, split `test`), about 14,000 four-option questions across 57 subjects. We sample 5,000–8,000 with a fixed seed. For RQ4, subjects map to MMLU's four categories: STEM, Humanities, Social Sciences, Other.

**Prompt.** This format is fixed for the whole project. Changing it invalidates comparisons.
```
Answer the following multiple choice question. Reply with only the letter (A, B, C, or D).

Question: {question}
A. {choice 0}
B. {choice 1}
C. {choice 2}
D. {choice 3}
```
It's wrapped in the model's chat template, ending right where the model would start its answer.

**Grading (automatic, no text generation):** read the model's next-token scores for `A`/`B`/`C`/`D`, pick the highest, and compare to the answer key. Confidence = softmax of those four scores, max value.

**Three stages of features.** We save the last token's hidden state at every layer:

| Stage | Input | Meaning |
|---|---|---|
| 1: question only | Question, no options | What the model "thinks" after reading the question |
| 2: question + options | The full grading prompt | Right before it answers |
| 3: after answer | Prompt + the chosen letter | After it commits |

**Sanity checks to expect:**
- Layer 0 at Stages 1–2 should score ≈ 0.5, because every question ends on the same template token.
- Layer 0 at Stage 3 can score above 0.5 because it encodes *which letter was picked*. That's a shortcut, so always report the chosen-letter baseline next to Stage 3.

**Extract once, share with everyone.** Features are saved to disk (`meta.csv` + one `.npy` per stage). Nobody should rerun the model to run experiments, since probes train in seconds from the saved files.

**Detectors we'll compare:** logistic regression (primary), logistic + PCA, small MLP, multi-layer MLP, and probe + confidence combined.

**Baselines we must report:** random, model confidence, entropy/margin, chosen-letter only, question length, subject only, and a text-only classifier (can the question text alone predict the mistakes?).

---

## 5. The Four Experiments

1. **Layer sweep (RQ1):** Train a probe per stage × layer and plot AUROC by layer with confidence bands. Say "layers ~X–Y are best," not "layer 17 is best," unless the error bars support it.
2. **Probes vs. confidence (RQ2):** Compare all detectors against the confidence baseline using a paired bootstrap. "Probes don't beat confidence" is also a valid finding.
3. **How early? (RQ3):** Compare best-layer AUROC across the three stages and against the text-only and subject-only baselines.
4. **Generalization (RQ4):** Train on one MMLU category and test on the others, producing a 4×4 heatmap. Use equal training sizes per category.

---

## 6. Ground Rules for Honest Results

1. **One fixed train/val/test split** (`splits.json`) for the whole team, with ≥1,000 test questions.
2. **Never tune on the test set.**
3. **Report 95% bootstrap confidence intervals** on every AUROC.
4. **Compare methods with paired tests** on the same test questions.
5. **Use ≥5 seeds** for anything random (MLPs).
6. **Always show baselines** next to results.
7. **Report negative results.** They count as findings.
8. **Log everything:** model, prompt version, seed, split file, library versions.

---

## 7. What You Need

- **Skills:** basic Python. No prior ML research or interpretability experience needed.
- **Hardware:** any 16 GB laptop or free Google Colab. No dedicated GPU required.
- **Disk:** ~7 GB for the model, plus ~3.6 GB for shared features at full scale.
- **Setup:**
  ```bash
  python -m venv venv && source venv/bin/activate
  pip install torch transformers datasets scikit-learn accelerate numpy matplotlib pandas
  ```
  Exact versions will be pinned in `requirements.txt`.

---

## 8. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Members get busy | Core result done by Week 4; experiments are independent |
| Probes don't beat confidence | Valid result; also try probe + confidence, more data, PCA |
| Noisy results | ≥1,000 test questions, bootstrap CIs, paired tests |
| Shortcuts inflate scores | Required shortcut baselines |
| Slow extraction | Run once overnight, split across laptops, or use Colab |
| Scope creep | Stretch goals only after Week 9 |

---

## 9. Stretch Goals (Second Semester)

- Model size comparison (0.5B / 1.5B / 3B / 7B)
- ICR Probe features (Zhang et al., 2025, the paper that inspired this project)
- Free-form "hallucination" version on TriviaQA
- Prompting effects (few-shot, "think step by step")
- Steering with a "right vs. wrong" direction
- Write-up for arXiv, a symposium, or a workshop

---

## 10. Deliverables

- Public GitHub repo with reproducible scripts and README
- Results CSVs and figures for all four experiments
- Written report (6–10 pages)
- Final presentation (15–20 min)

---

## 11. FAQ

**Do we train the AI model?** No. We only train small probes on its hidden states.

**Where's the ML, then?** The probe is a supervised binary classifier. The skills are feature selection, model comparison, avoiding overfitting, tuning, proper splits, evaluation, baselines, and generalization tests.

**Can we use ChatGPT/Claude APIs?** No. Closed models don't expose hidden states, so we need an open-weight model.

**Why MMLU instead of a hallucination dataset?** Multiple choice gives automatic, noise-free grading. Free-form is a stretch goal.

**Isn't "the model knows before answering" already known?** Partly (Kadavath et al. 2022; Kossen et al. 2024). We measure it in our setup against text-only baselines and won't claim it's new.

**What counts as success?** Complete, reproducible answers to RQ1–RQ4 with honest uncertainty, a clean repo, and a clear presentation.

---

## References

- Azaria, A., & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* Findings of EMNLP 2023.
- Hendrycks, D., et al. (2021). *Measuring Massive Multitask Language Understanding.* ICLR 2021.
- Kadavath, S., et al. (2022). *Language Models (Mostly) Know What They Know.* arXiv:2207.05221.
- Kossen, J., et al. (2024). *Semantic Entropy Probes.* arXiv:2406.15927.
- Orgad, H., et al. (2024). *LLMs Know More Than They Show.* arXiv:2410.02707.
- Zhang, Z., et al. (2025). *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs.* ACL 2025.
