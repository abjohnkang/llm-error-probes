# llm-error-probes

Do AI models know when they're wrong? We give Qwen2.5-3B-Instruct multiple-choice questions from MMLU, record its hidden states at every layer, and train small classifiers (probes) to predict when its answer will be wrong. The model itself is never trained or changed.

DAIS Club research project. Full plan: [DAIS_PROJECT_OVERVIEW.md](DAIS_PROJECT_OVERVIEW.md)

## Requirements

- macOS on Apple Silicon (M1 or later). The scripts run the model on the Mac GPU (MPS).
- 16 GB RAM
- Python 3.14
- About 7 GB of free disk space for the model

## Setup

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/abjohnkang/llm-error-probes.git
cd llm-error-probes
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

One-time download of the model (~6 GB) and the MMLU questions:

```bash
HF_HUB_OFFLINE=0 python scripts/one_question.py
```

After this, the scripts run offline using the downloaded files.

Each time you open a new terminal, activate the environment again:

```bash
cd ~/src/llm-error-probes
source .venv/bin/activate
```

## Run

Ask the model one question and see its A/B/C/D probabilities:

```bash
python scripts/one_question.py         # random question
python scripts/one_question.py 1234    # a specific question
```

Run the pilot: grade the model, extract hidden states, and train a probe for each layer:

```bash
python scripts/pilot.py Qwen/Qwen2.5-3B-Instruct 50     # ~1.5 min, quick test
python scripts/pilot.py Qwen/Qwen2.5-3B-Instruct 400    # ~25 min, more reliable
```

AUROC: 0.5 = coin flip, 1.0 = perfect. The baseline to beat is the model's own confidence.

## How we work

Never push directly to `main`. Every change goes through a pull request.

```bash
git checkout main
git pull                                  # start from the latest main
git checkout -b <short-description>       # make your own branch
# ...edit files...
git add <files>
git commit -m "Describe the change"
git push -u origin <short-description>
gh pr create --fill                       # open a pull request
```

After the PR is reviewed and merged on GitHub:

```bash
git checkout main
git pull
```

## Repo layout

- `scripts/`: `one_question.py` (demo) and `pilot.py` (proof of concept for the full pipeline)
- `results/predicted_errors/`: each member's predicted model mistake (see `EXAMPLE.md`)
- `requirements.txt`: exact package versions
