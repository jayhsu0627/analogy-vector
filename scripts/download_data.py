"""Download the train/dev/test prompt pairs from the Hugging Face Hub into ./data.

    python scripts/download_data.py
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download

from analogy_vector.data import DEFAULT_DATA_DIR, HF_DATASET_ID, SPLITS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_id", default=HF_DATASET_ID)
    p.add_argument("--out_dir", default=DEFAULT_DATA_DIR)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in SPLITS:
        cached = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=f"data/{split}.jsonl")
        dest = os.path.join(args.out_dir, f"{split}.jsonl")
        shutil.copyfile(cached, dest)
        with open(dest, encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        print(f"{split:5s}: {n} pairs -> {dest}")


if __name__ == "__main__":
    main()
