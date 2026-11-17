"""Pick the steering layer l* from the dev sweep: argmax HM among layers with KL < 0.1.

    python evaluation/select_layer.py

Reads results/layer_sweep/gens_<model>__summary.json (and results/layer_stability.json if present,
to report the gap to the stability transition layer). Writes results/layer_selection.json, which
evaluation/generate_test.py accepts via --selection.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import KL_THRESHOLD, RESULTS_DIR, read_json, write_json
from analogy_vector.models import load_config, safe_name


def best_layer(summary: list, kl_threshold: float = KL_THRESHOLD):
    candidates = [s for s in summary if s["avg_kl"] < kl_threshold]
    if not candidates:
        print(f"  warning: no layer has KL < {kl_threshold}; using argmax HM over all layers")
        candidates = summary
    best = max(candidates, key=lambda s: s["mean_hm"])
    return best["layer"], best["mean_hm"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", default=os.path.join(RESULTS_DIR, "layer_sweep"))
    p.add_argument("--stability", default=os.path.join(RESULTS_DIR, "layer_stability.json"))
    p.add_argument("--kl_threshold", type=float, default=KL_THRESHOLD)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "layer_selection.json"))
    args = p.parse_args()

    stability = read_json(args.stability) if os.path.exists(args.stability) else {}
    selection, pred, emp = {}, [], []
    print(f"{'model':24s} {'l*':>4s} {'dev HM':>7s} {'stability':>9s}")
    for model_key in load_config():
        path = os.path.join(args.sweep_dir, f"gens_{safe_name(model_key)}__summary.json")
        if not os.path.exists(path):
            continue
        layer, hm = best_layer(read_json(path), args.kl_threshold)
        transition = stability.get(model_key, {}).get("transition_layer")
        selection[model_key] = {"empirical_layer": layer, "empirical_hm": hm, "predicted_layer": transition,
                                "gap": None if transition is None else transition - layer}
        print(f"{model_key:24s} {layer:4d} {hm:7.2f} {str(transition):>9s}")
        if transition is not None:
            pred.append(transition)
            emp.append(layer)

    if len(pred) > 1:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(pred, emp)
        gap = float(np.mean([abs(a - b) for a, b in zip(pred, emp)]))
        selection["overall"] = {"spearman_rho": rho, "spearman_p": pval, "mean_abs_gap": gap, "n_models": len(pred)}
        print(f"Spearman(stability, empirical) = {rho:.3f} (p={pval:.3f}); mean |gap| = {gap:.2f} layers")
    write_json(args.out, selection)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
