"""Generate test-set responses for the three conditions of Table 2.

  unsteered_literal   literal prompt, no steering
  prompting_baseline  "Explain <concept> using an analogy.", no steering
  steered_clean       literal prompt + alpha * v_clean at layer l*, for each alpha in --alphas

    python evaluation/generate_test.py --model all
    python evaluation/judge.py --run_dir results/test_eval/final
    python evaluation/summarize.py --run_dir results/test_eval/final

Layers come from configs/models.json, or from a select_layer.py output via --selection.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm

from common import RESULTS_DIR, batched, ensure_dir, read_json, silence_progress_bars, write_json, write_jsonl
from analogy_vector.data import DEFAULT_DATA_DIR, load_split
from analogy_vector.models import VECTORS_DIR, load_config, load_model, load_vector
from analogy_vector.steering import generate, generate_with_steering


def prompting_baseline_prompt(concept: str) -> str:
    return f"Explain {concept} using an analogy."


def main() -> None:
    silence_progress_bars()
    config = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(config) + ["all"])
    p.add_argument("--selection", default=None, help="layer_selection.json from select_layer.py (default: config)")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--vectors_dir", default=VECTORS_DIR)
    p.add_argument("--split", default="test")
    p.add_argument("--run_id", default="final")
    p.add_argument("--alphas", default="0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    selection = read_json(args.selection) if args.selection else None
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    examples = load_split(args.split, args.data_dir)[: args.limit] if args.limit else load_split(args.split, args.data_dir)
    model_keys = list(config) if args.model == "all" else [args.model]
    out_dir = ensure_dir(os.path.join(RESULTS_DIR, "test_eval", args.run_id))
    gens_dir = ensure_dir(os.path.join(out_dir, "generations"))
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

    for model_key in model_keys:
        layer = selection[model_key]["empirical_layer"] if selection else config[model_key]["layer"]
        print(f"\n=== {model_key} (layer {layer}) ===")
        model = None

        def get_model():
            nonlocal model
            if model is None:
                model = load_model(config[model_key]["hf_id"])
            return model

        def row(ex, condition, alpha, prompt, text, **extra):
            return {"model_key": model_key, "condition": condition, "alpha": alpha, **extra,
                    "concept_id": ex.concept_id, "domain": ex.domain, "prompt": prompt, "generation": text}

        for condition, make_prompt in (("unsteered_literal", lambda ex: ex.literal_prompt),
                                       ("prompting_baseline", lambda ex: prompting_baseline_prompt(ex.concept))):
            path = os.path.join(gens_dir, f"{model_key}__{condition}.jsonl")
            if os.path.exists(path):
                continue
            rows = []
            for batch in tqdm(batched(examples, args.batch_size), desc=condition):
                prompts = [make_prompt(ex) for ex in batch]
                texts = generate(get_model(), prompts, **gen_kwargs)
                rows.extend(row(ex, condition, 0.0, pr, t) for ex, pr, t in zip(batch, prompts, texts))
            write_jsonl(path, rows)

        path = os.path.join(gens_dir, f"{model_key}__steered_clean.jsonl")
        if not os.path.exists(path):
            layer_vec = load_vector(model_key, clean=True, vectors_dir=args.vectors_dir)[layer]
            rows = []
            for alpha in alphas:
                for batch in tqdm(batched(examples, args.batch_size), desc=f"steered alpha={alpha}"):
                    prompts = [ex.literal_prompt for ex in batch]
                    texts = generate_with_steering(get_model(), prompts, layer_vec, layer, alpha, **gen_kwargs)
                    rows.extend(row(ex, "steered_clean", alpha, pr, t, target_layer=layer)
                                for ex, pr, t in zip(batch, prompts, texts))
            write_jsonl(path, rows)

        del model
        torch.cuda.empty_cache()

    write_json(os.path.join(out_dir, "run_config.json"),
               {"models": model_keys, "alphas": alphas, "split": args.split, "run_id": args.run_id})
    print(f"\nDone. Generations in: {gens_dir}")


if __name__ == "__main__":
    main()
