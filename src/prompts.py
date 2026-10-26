"""The fixed project prompts for the 3 stages (DAIS_PROJECT_OVERVIEW.md, Section 4).

Keep these fixed for the whole project. Changing a prompt changes the model's
behavior, so results from before and after the change can't be compared.
If you ever change them, bump PROMPT_VERSION.
"""

PROMPT_VERSION = "v1"
LETTERS = ["A", "B", "C", "D"]
INSTRUCTION = "Answer the following multiple choice question. Reply with only the letter (A, B, C, or D).\n\n"


def chat(tok, user, assistant_prefix=""):
    """Wrap text in the model's chat format, as if a user sent it.
    assistant_prefix pre-fills the start of the model's reply (used in stage 3)."""
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + assistant_prefix


def options_text(ex):
    return "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, ex["choices"]))


def stage1(tok, ex):
    """Question only: no instruction, no options."""
    return chat(tok, f"Question: {ex['question']}")


def stage2(tok, ex):
    """The full grading prompt: instruction + question + options."""
    return chat(tok, INSTRUCTION + f"Question: {ex['question']}\n{options_text(ex)}")


def stage3(tok, ex, pick):
    """The grading prompt with the model's chosen letter appended as its reply."""
    return chat(tok, INSTRUCTION + f"Question: {ex['question']}\n{options_text(ex)}", LETTERS[pick])
