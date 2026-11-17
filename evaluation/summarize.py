"""Aggregate judged test generations into tables (mean / std / SEM of AQ, FL, IF, HM).

    python evaluation/summarize.py --run_dir results/test_eval/final

Writes <run_dir>/tables/aggregate_by_model_condition_alpha.json (Table 2: take the best alpha per
model for steered_clean), aggregate_by_model_condition_alpha_domain.json, aggregate_by_model_condition.json.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import ensure_dir, read_jsonl, write_json

METRICS = ("analogy_quality", "fluency_coherence", "instruction_following", "harmonic_mean")


def agg(rows):
    out = {"n": float(len(rows))}
    for k in METRICS:
        vs = [float(r["judge"][k]) for r in rows]
        out[f"mean_{k}"] = float(np.mean(vs))
        out[f"std_{k}"] = float(np.std(vs))
        out[f"sem_{k}"] = float(np.std(vs) / max(1.0, np.sqrt(len(vs))))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "judge_three_metric", "*__scored.jsonl")))
    if not files:
        raise FileNotFoundError(f"No scored files under {args.run_dir}/judge_three_metric")
    rows = [r for f in files for r in read_jsonl(f)]

    by_mca, by_mcad, by_mc = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        m, c, a, d = r["model_key"], r["condition"], float(r.get("alpha", 0.0)), r.get("domain", "unknown")
        by_mca[(m, c, a)].append(r)
        by_mcad[(m, c, a, d)].append(r)
        by_mc[(m, c)].append(r)

    tables = ensure_dir(os.path.join(args.run_dir, "tables"))
    write_json(os.path.join(tables, "aggregate_by_model_condition_alpha.json"),
               [{"model_key": m, "condition": c, "alpha": a, **agg(v)} for (m, c, a), v in sorted(by_mca.items())])
    write_json(os.path.join(tables, "aggregate_by_model_condition_alpha_domain.json"),
               [{"model_key": m, "condition": c, "alpha": a, "domain": d, **agg(v)}
                for (m, c, a, d), v in sorted(by_mcad.items())])
    write_json(os.path.join(tables, "aggregate_by_model_condition.json"),
               [{"model_key": m, "condition": c, **agg(v)} for (m, c), v in sorted(by_mc.items())])

    print(f"{'model':24s} {'condition':20s} {'alpha':>5s} {'AQ':>5s} {'FL':>5s} {'IF':>5s} {'HM':>5s}")
    for (m, c, a), v in sorted(by_mca.items()):
        s = agg(v)
        print(f"{m:24s} {c:20s} {a:5.2f} {s['mean_analogy_quality']:5.2f} {s['mean_fluency_coherence']:5.2f} "
              f"{s['mean_instruction_following']:5.2f} {s['mean_harmonic_mean']:5.2f}")
    print(f"Tables in: {tables}")


if __name__ == "__main__":
    main()
