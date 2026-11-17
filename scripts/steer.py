"""Steer a model toward (alpha > 0) or away from (alpha < 0) analogy-based explanations.

    python scripts/steer.py --model gemma-2-2b-it --prompt "Explain how a refrigerator works."
    python scripts/steer.py --model qwen-2.5-7b-instruct --alpha 2.0 --prompt "..."

Defaults use the paper's selected layer and best alpha (configs/models.json) and the clean vector.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from analogy_vector.models import VECTORS_DIR, load_config, load_model, load_vector
from analogy_vector.steering import generate, generate_with_steering


def main() -> None:
    config = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(config))
    p.add_argument("--prompt", required=True, action="append", help="Repeat to steer several prompts in one batch")
    p.add_argument("--alpha", type=float, default=None, help="Steering coefficient (default: paper's best alpha)")
    p.add_argument("--layer", type=int, default=None, help="Steering layer (default: paper's selected layer)")
    p.add_argument("--raw", action="store_true", help="Use the raw (non-orthogonalized) vector")
    p.add_argument("--vectors_dir", default=VECTORS_DIR)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_baseline", action="store_true", help="Skip the unsteered generation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    spec = config[args.model]
    layer = spec["layer"] if args.layer is None else args.layer
    alpha = spec["alpha"] if args.alpha is None else args.alpha
    vector = load_vector(args.model, clean=not args.raw, vectors_dir=args.vectors_dir)[layer]

    model = load_model(spec["hf_id"], args.device, args.dtype)
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

    if not args.no_baseline:
        torch.manual_seed(args.seed)
        baseline = generate(model, args.prompt, **gen_kwargs)
    torch.manual_seed(args.seed)
    steered = generate_with_steering(model, args.prompt, vector, layer, alpha, **gen_kwargs)

    kind = "raw" if args.raw else "clean"
    for i, prompt in enumerate(args.prompt):
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        if not args.no_baseline:
            print("-" * 80 + "\nUNSTEERED (alpha = 0):\n" + baseline[i])
        print("-" * 80 + f"\nSTEERED ({kind} vector, layer {layer}, alpha = {alpha}):\n" + steered[i])


if __name__ == "__main__":
    main()
