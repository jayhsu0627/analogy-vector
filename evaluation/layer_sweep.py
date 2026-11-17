"""Layer sweep on the dev split (Section 4.1 / Appendix A.2).

For every layer l: steer the literal dev prompts with alpha * v_l (raw vector), record the
next-token KL(unsteered || steered) on the first --kl_samples prompts, generate, then score all
generations with the three-metric judge.

    python evaluation/layer_sweep.py --model gemma-2-2b-it
    python evaluation/layer_sweep.py --model all --skip_judge   # generations only

Writes results/layer_sweep/gens_<model>.jsonl, gens_<model>__scored.jsonl, gens_<model>__summary.json
(per layer: mean_hm, avg_kl). Then run evaluation/select_layer.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from common import RESULTS_DIR, ensure_dir, read_jsonl, silence_progress_bars, write_json, write_jsonl
from analogy_vector.data import DEFAULT_DATA_DIR, load_split
from analogy_vector.models import VECTORS_DIR, load_config, load_model, load_vector, safe_name
from analogy_vector.steering import generate_with_steering, kl_divergence

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"


def run_judge_on_gens(gens_file: str, judge_model: str, gpu_memory_utilization: float = 0.9,
                      batch_size: int = 64) -> None:
    from judge import judge_batch, load_judge

    scored_file = gens_file.replace(".jsonl", "__scored.jsonl")
    summary_file = gens_file.replace(".jsonl", "__summary.json")
    if os.path.exists(scored_file) and os.path.exists(summary_file):
        print(f"Judged results for {os.path.basename(gens_file)} exist, skipping.")
        return

    judge_llm = load_judge(judge_model, gpu_memory_utilization)
    rows = read_jsonl(gens_file)
    scored = []
    for i in tqdm(range(0, len(rows), batch_size), desc="Judging"):
        batch = rows[i : i + batch_size]
        scores = judge_batch(judge_llm, [{"prompt": r["prompt"], "generation": r["generation"]} for r in batch])
        scored.extend({**r, "judge": s} for r, s in zip(batch, scores))
    write_jsonl(scored_file, scored)

    per_layer = {}
    for r in scored:
        per_layer.setdefault(r["layer"], {"hms": [], "kl": r["avg_kl"]})["hms"].append(r["judge"]["harmonic_mean"])
    summary = [{"layer": l, "mean_hm": float(np.mean(d["hms"])), "avg_kl": d["kl"]} for l, d in sorted(per_layer.items())]
    write_json(summary_file, summary)
    print(f"Layer-wise summary -> {summary_file}")

    del judge_llm
    torch.cuda.empty_cache()


def main() -> None:
    silence_progress_bars()
    config = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(config) + ["all"])
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--vectors_dir", default=VECTORS_DIR)
    p.add_argument("--results_dir", default=os.path.join(RESULTS_DIR, "layer_sweep"))
    p.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit_dev", type=int, default=100)
    p.add_argument("--kl_samples", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--layers", default=None, help="Comma-separated subset of layers (default: all)")
    p.add_argument("--skip_judge", action="store_true")
    args = p.parse_args()

    ensure_dir(args.results_dir)
    dev = load_split("dev", args.data_dir)[: args.limit_dev]
    model_keys = list(config) if args.model == "all" else [args.model]

    for model_key in model_keys:
        gens_file = os.path.join(args.results_dir, f"gens_{safe_name(model_key)}.jsonl")
        if os.path.exists(gens_file):
            print(f"Generations for {model_key} exist.")
        else:
            model = load_model(config[model_key]["hf_id"])
            vector = load_vector(model_key, clean=False, vectors_dir=args.vectors_dir, device=model.cfg.device)
            layers = [int(x) for x in args.layers.split(",")] if args.layers else range(model.cfg.n_layers)
            kl_prompts = [ex.literal_prompt for ex in dev[: args.kl_samples]]
            rows = []
            for layer in layers:
                layer_vec = vector[layer]
                avg_kl = kl_divergence(model, kl_prompts, layer, layer_vec, args.alpha)
                for i in tqdm(range(0, len(dev), args.batch_size), desc=f"{model_key} layer {layer}", leave=False):
                    batch = dev[i : i + args.batch_size]
                    texts = generate_with_steering(
                        model, [ex.literal_prompt for ex in batch], layer_vec, layer, args.alpha,
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                    )
                    rows.extend({"model_key": model_key, "layer": layer, "concept_id": ex.concept_id,
                                 "prompt": ex.literal_prompt, "generation": text, "avg_kl": avg_kl}
                                for ex, text in zip(batch, texts))
            write_jsonl(gens_file, rows)
            del model
            torch.cuda.empty_cache()

        if not args.skip_judge:
            run_judge_on_gens(gens_file, args.judge_model)


if __name__ == "__main__":
    main()
