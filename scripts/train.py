"""Compute the raw and orthogonalized ("clean") analogy vectors for one or more models.

    python scripts/train.py --model gemma-2-2b-it
    python scripts/train.py --model all

Writes analogy_vector_<model>.pt (raw), analogy_vector_<model>_clean.pt, and
analogy_vector_<model>_clean_analysis.json (per-layer, per-token removed fraction) to --out_dir.
Both .pt files hold a [n_layers, d_model] tensor.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from analogy_vector.data import DEFAULT_DATA_DIR, REPO_ROOT, load_split
from analogy_vector.models import load_config, load_model, vector_path
from analogy_vector.vector import DEFAULT_TOKENS, compute_analogy_vector, orthogonalize


def main() -> None:
    config = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(config) + ["all"])
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "runs", "vectors"))
    p.add_argument("--tokens", default=None, help="Comma-separated trigger tokens (default: the 16 in vector.py)")
    p.add_argument("--limit", type=int, default=None, help="Use only the first N training pairs (debugging)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--force", action="store_true", help="Overwrite existing vectors")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    examples = load_split("train", args.data_dir)
    if args.limit:
        examples = examples[: args.limit]
    tokens = [t for t in args.tokens.split(",")] if args.tokens else DEFAULT_TOKENS
    model_keys = list(config) if args.model == "all" else [args.model]

    for model_key in model_keys:
        raw_path = vector_path(model_key, clean=False, vectors_dir=args.out_dir)
        clean_path = vector_path(model_key, clean=True, vectors_dir=args.out_dir)
        if os.path.exists(clean_path) and not args.force:
            print(f"{clean_path} exists, skipping {model_key} (use --force to recompute).")
            continue

        print(f"=== {model_key}: {len(examples)} training pairs ===")
        model = load_model(config[model_key]["hf_id"], args.device, args.dtype)

        raw = compute_analogy_vector(model, examples)
        torch.save(raw, raw_path)
        print(f"Raw vector {tuple(raw.shape)} -> {raw_path}")

        clean, stats = orthogonalize(raw, model, tokens)
        torch.save(clean.cpu(), clean_path)
        layer = config[model_key]["layer"]
        reduction = 1 - (clean[layer].float().norm() / raw[layer].float().norm()).item()
        print(f"Clean vector -> {clean_path} (layer {layer}: norm reduced by {reduction * 100:.2f}%)")

        with open(clean_path.replace(".pt", "_analysis.json"), "w") as f:
            json.dump({"model": model_key, "tokens_used": tokens,
                       "layer_stats": {i: s for i, s in enumerate(stats)}}, f, indent=2)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
