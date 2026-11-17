"""Paired literal/analogy prompts built from ELI5 concepts (602 train / 99 dev / 299 test)."""

import dataclasses
import json
import os
from typing import List

HF_DATASET_ID = "HuggingJady/analogy-vector"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
SPLITS = ("train", "dev", "test")


@dataclasses.dataclass(frozen=True)
class Example:
    concept_id: str
    concept: str
    domain: str
    literal_prompt: str
    analogy_prompt: str


def load_split(split: str, data_dir: str = DEFAULT_DATA_DIR) -> List[Example]:
    """Load one split in its original order (the sweep and KL use the first dev prompts)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            examples.append(
                Example(
                    concept_id=r["concept_id"],
                    concept=r["concept"],
                    domain=r.get("domain", "unknown"),
                    literal_prompt=r["literal_prompt"],
                    analogy_prompt=r["analogy_prompt"],
                )
            )
    return examples
