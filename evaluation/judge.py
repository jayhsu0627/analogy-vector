"""Three-metric LLM judge (Qwen2.5-72B-Instruct-AWQ via vLLM).

Scores each response on Analogy Quality, Fluency/Coherence, and Instruction-Following (1-5 each)
and reports their harmonic mean (HM), following AxBench.

    python evaluation/judge.py --run_dir results/test_eval/<run_id>

Reads <run_dir>/generations/*.jsonl, writes <run_dir>/judge_three_metric/*__scored.jsonl + summary.json.
"""

import argparse
import glob
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tqdm import tqdm

from common import (
    clamp_int,
    ensure_dir,
    extract_first_json_obj,
    harmonic_mean,
    read_json,
    read_jsonl,
    silence_progress_bars,
    write_json,
    write_jsonl,
)

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"


def build_judge_prompt(user_prompt: str, model_response: str) -> str:
    return f"""You are an expert evaluator of explanation quality.

You will be given:
1) The user's prompt
2) A model response

Rate the response on THREE independent 1–5 scales:

1) Analogy Quality (1–5):
- 1: No analogy/metaphor OR a clearly incorrect/confusing analogy
- 3: Some analogy present, partially helpful but shallow or imperfect
- 5: Clear, helpful, well-developed analogy with correct structural correspondence

2) Fluency / Coherence (1–5):
- 1: Unreadable, incoherent, repetitive, or severely ungrammatical
- 3: Mostly understandable with some issues
- 5: Very fluent, coherent, well-structured

3) Instruction-Following / Correctness relative to the question (1–5):
- 1: Does not answer the prompt or is largely incorrect
- 3: Partially answers with some inaccuracies or misses key parts
- 5: Fully answers and is correct relative to the prompt

Important:
- Judge each dimension independently.
- Do NOT reward keyword stuffing.
- Output JSON ONLY in the following schema:
{{
  "analogy_quality": 1,
  "fluency_coherence": 1,
  "instruction_following": 1,
  "rationale": "brief, 1-3 sentences"
}}

Prompt:
{user_prompt}

Response:
{model_response}
"""


def load_judge(model_name: str = DEFAULT_JUDGE_MODEL, gpu_memory_utilization: float = 0.90,
               tensor_parallel_size: int = 1):
    from vllm import LLM

    print(f"Loading judge model: {model_name}")
    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
    )


@contextmanager
def _quiet_stderr():
    old = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old


def judge_batch(judge_llm, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """items: [{"prompt": ..., "generation": ...}] -> judge dicts (greedy, max 256 tokens)."""
    from vllm import SamplingParams

    prompts = [build_judge_prompt(i["prompt"], i["generation"]) for i in items]
    sampling = SamplingParams(temperature=0.0, max_tokens=256)
    with _quiet_stderr():
        outs = judge_llm.generate(prompts, sampling)
    results = []
    for out in outs:
        text = out.outputs[0].text.strip()
        obj = extract_first_json_obj(text) or {}
        aq = clamp_int(obj.get("analogy_quality"), 1, 5, default=3)
        fl = clamp_int(obj.get("fluency_coherence"), 1, 5, default=3)
        inst = clamp_int(obj.get("instruction_following"), 1, 5, default=3)
        rationale = obj.get("rationale", "")
        results.append({
            "analogy_quality": aq,
            "fluency_coherence": fl,
            "instruction_following": inst,
            "harmonic_mean": harmonic_mean([aq, fl, inst]),
            "rationale": rationale if isinstance(rationale, str) else "",
            "raw_judge_text": text,
        })
    return results


def main() -> None:
    silence_progress_bars()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--glob", default="*.jsonl", help="Which files in generations/ to score")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Score only the first N rows per file (debugging)")
    args = p.parse_args()

    gens_dir = os.path.join(args.run_dir, "generations")
    out_dir = ensure_dir(os.path.join(args.run_dir, "judge_three_metric"))
    run_cfg = read_json(os.path.join(args.run_dir, "run_config.json"))

    gen_files = sorted(glob.glob(os.path.join(gens_dir, args.glob)))
    if not gen_files:
        raise FileNotFoundError(f"No generation files matched {args.glob} in {gens_dir}")
    file_rows = {gf: read_jsonl(gf)[: args.limit] if args.limit else read_jsonl(gf) for gf in gen_files}

    judge_llm = load_judge(args.judge_model, args.gpu_memory_utilization, args.tensor_parallel_size)
    pbar = tqdm(total=sum(len(r) for r in file_rows.values()), desc="Judging", unit="row")
    summaries: Dict[str, Any] = {}
    for gf, rows in file_rows.items():
        scored = []
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            scores = judge_batch(judge_llm, [{"prompt": r["prompt"], "generation": r["generation"]} for r in batch])
            scored.extend({**r, "judge": s} for r, s in zip(batch, scores))
            pbar.update(len(batch))
        write_jsonl(os.path.join(out_dir, os.path.basename(gf).replace(".jsonl", "__scored.jsonl")), scored)

        col = lambda k: [s["judge"][k] for s in scored]
        summaries[os.path.basename(gf)] = {
            "file": os.path.basename(gf),
            "n": len(scored),
            "mean_harmonic_mean": float(np.mean(col("harmonic_mean"))) if scored else float("nan"),
            "mean_analogy_quality": float(np.mean(col("analogy_quality"))) if scored else float("nan"),
            "mean_fluency_coherence": float(np.mean(col("fluency_coherence"))) if scored else float("nan"),
            "mean_instruction_following": float(np.mean(col("instruction_following"))) if scored else float("nan"),
            "std_harmonic_mean": float(np.std(col("harmonic_mean"))) if scored else float("nan"),
        }
    pbar.close()
    write_json(os.path.join(out_dir, "summary.json"), {"run_config": run_cfg, "summaries": summaries})
    print(f"Done. Scored outputs in: {out_dir}")


if __name__ == "__main__":
    main()
