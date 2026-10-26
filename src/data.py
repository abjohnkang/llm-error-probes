"""Load MMLU, sample questions with a fixed seed, and map subjects to categories.

Each question keeps its MMLU test-set index as its id ("qid"), so the same
question always has the same id, no matter how many questions we sample.
"""
import random
from datasets import load_dataset

# ---------------------------------------------------------------------------
# MMLU's 57 subjects -> 4 categories (STEM, Humanities, Social Sciences, Other).
# Copied from the original MMLU repo (hendrycks/test, categories.py).
# ---------------------------------------------------------------------------
SUBCATEGORY = {
    "abstract_algebra": "math", "anatomy": "health", "astronomy": "physics",
    "business_ethics": "business", "clinical_knowledge": "health",
    "college_biology": "biology", "college_chemistry": "chemistry",
    "college_computer_science": "computer science", "college_mathematics": "math",
    "college_medicine": "health", "college_physics": "physics",
    "computer_security": "computer science", "conceptual_physics": "physics",
    "econometrics": "economics", "electrical_engineering": "engineering",
    "elementary_mathematics": "math", "formal_logic": "philosophy",
    "global_facts": "other", "high_school_biology": "biology",
    "high_school_chemistry": "chemistry", "high_school_computer_science": "computer science",
    "high_school_european_history": "history", "high_school_geography": "geography",
    "high_school_government_and_politics": "politics", "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math", "high_school_microeconomics": "economics",
    "high_school_physics": "physics", "high_school_psychology": "psychology",
    "high_school_statistics": "math", "high_school_us_history": "history",
    "high_school_world_history": "history", "human_aging": "health",
    "human_sexuality": "culture", "international_law": "law", "jurisprudence": "law",
    "logical_fallacies": "philosophy", "machine_learning": "computer science",
    "management": "business", "marketing": "business", "medical_genetics": "health",
    "miscellaneous": "other", "moral_disputes": "philosophy", "moral_scenarios": "philosophy",
    "nutrition": "health", "philosophy": "philosophy", "prehistory": "history",
    "professional_accounting": "other", "professional_law": "law",
    "professional_medicine": "health", "professional_psychology": "psychology",
    "public_relations": "politics", "security_studies": "politics", "sociology": "culture",
    "us_foreign_policy": "politics", "virology": "health", "world_religions": "philosophy",
}
CATEGORY = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "Humanities": ["history", "philosophy", "law"],
    "Social Sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "Other": ["other", "business", "health"],
}
SUBCAT_TO_CATEGORY = {sub: cat for cat, subs in CATEGORY.items() for sub in subs}


def category_of(subject):
    return SUBCAT_TO_CATEGORY[SUBCATEGORY[subject]]


def load_mmlu():
    return load_dataset("cais/mmlu", "all", split="test")


def sample_qids(ds, n, seed=0):
    """Pick n random question ids. Same seed -> same questions every time."""
    return random.Random(seed).sample(range(len(ds)), n)
