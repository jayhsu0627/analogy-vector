"""Inter-layer cosine stability of the raw analogy vector (Appendix A.2, "Stability Transition").

The transition layer is the first layer in the second half of the model whose cosine similarity
with the next layer's vector exceeds 0.90. No model or GPU is needed.

    python evaluation/stability.py

Writes results/layer_stability.json.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import RESULTS_DIR, write_json
from analogy_vector.models import VECTORS_DIR, load_config, load_vector

THRESHOLD = 0.90


def cosine_stability(vector: torch.Tensor, threshold: float = THRESHOLD) -> dict:
    vector = vector.float()
    n_layers = vector.shape[0]
    sims = [
        torch.nn.functional.cosine_similarity(vector[l].unsqueeze(0), vector[l + 1].unsqueeze(0)).item()
        for l in range(n_layers - 1)
    ]
    # Search only the second half to skip early spikes (e.g. Gemma-2-9B at layer 1).
    transition = next((l for l in range(n_layers // 2, len(sims)) if sims[l] > threshold), None)
    return {"n_layers": n_layers, "similarities": sims, "transition_layer": transition}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vectors_dir", default=VECTORS_DIR)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "layer_stability.json"))
    args = p.parse_args()

    results = {}
    for model_key in load_config():
        results[model_key] = cosine_stability(load_vector(model_key, clean=False, vectors_dir=args.vectors_dir))
        print(f"{model_key:24s} transition layer: {results[model_key]['transition_layer']}")
    write_json(args.out, results)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
