"""Control experiments (Section 4.4 / Appendix A.5) plus the clean-vs-raw comparison (Appendix B).

  1. random      random vector with the same norm as v_l, alpha in [0, 2]   (specificity)
  2. simplicity  steer a creative-writing prompt; track analogy keywords and word count
  3. negative    alpha in {-1.0, -0.5, ..., 2.0} on a literal prompt          (bidirectionality)
  4. clean_raw   raw vs clean vector up to alpha = 6; count "imagine" (mode collapse)

Tests 1-3 steer with the raw vector, as in the original runs. Metrics are analogy-keyword counts
on single greedy-ish generations (80 new tokens, temperature 0.1).

    python evaluation/controls.py --model gemma-2-9b-it

Writes results/controls_<model>_layer<l>.json.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import RESULTS_DIR, write_json
from analogy_vector.models import VECTORS_DIR, format_chat, load_config, load_model, load_vector
from analogy_vector.steering import hook_name

TECHNICAL_PROMPT = "Explain how TCP/IP handshakes work."
POETRY_PROMPT = "Write a short poem about the ocean."
LITERAL_PROMPT = "Explain DNS technically."
KEYWORDS = ["like", "imagine", "similar", "analogy", "metaphor", "think of", "as if"]

POSITIVE_COEFFS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
NEGATIVE_COEFFS = [-1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
COLLAPSE_COEFFS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def count_keywords(text: str) -> int:
    """Number of distinct analogy keywords present (proxy metric)."""
    lower = text.lower()
    return sum(1 for kw in KEYWORDS if kw in lower)


@torch.no_grad()
def steer_once(model, prompt: str, vector: torch.Tensor, layer: int, coeff: float,
               max_new_tokens: int = 80, temperature: float = 0.1) -> str:
    formatted = format_chat(prompt, model)
    tokens = model.to_tokens(formatted, prepend_bos=False).to(model.cfg.device)

    def hook(resid_post, hook):
        resid_post += coeff * vector
        return resid_post

    with model.hooks(fwd_hooks=[(hook_name(layer), hook)]):
        ids = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=temperature, verbose=False)
    text = model.tokenizer.decode(ids[0])
    return text[len(formatted):].strip() if text.startswith(formatted) else text


def test_random(model, vec, layer):
    random_vec = torch.randn(vec.shape[0], device=vec.device, dtype=vec.dtype)
    random_vec = random_vec / random_vec.norm() * vec.norm()
    out = {"analogy_norm": vec.norm().item(), "analogy": [], "random": []}
    for c in POSITIVE_COEFFS:
        for name, v in (("analogy", vec), ("random", random_vec)):
            r = steer_once(model, TECHNICAL_PROMPT, v, layer, c)
            out[name].append({"coeff": c, "keywords": count_keywords(r), "response": r})
        print(f"[random]     alpha={c:4.2f}  analogy kw={out['analogy'][-1]['keywords']}  "
              f"random kw={out['random'][-1]['keywords']}")
    return out


def test_simplicity(model, vec, layer):
    out = []
    for c in POSITIVE_COEFFS:
        r = steer_once(model, POETRY_PROMPT, vec, layer, c)
        out.append({"coeff": c, "keywords": count_keywords(r), "word_count": len(r.split()), "response": r})
        print(f"[simplicity] alpha={c:4.2f}  kw={out[-1]['keywords']}  words={out[-1]['word_count']}")
    return out


def test_negative(model, vec, layer):
    out = []
    for c in NEGATIVE_COEFFS:
        r = steer_once(model, LITERAL_PROMPT, vec, layer, c)
        out.append({"coeff": c, "keywords": count_keywords(r), "response": r})
        print(f"[negative]   alpha={c:5.2f}  kw={out[-1]['keywords']}")
    return out


def test_clean_vs_raw(model, raw_vec, clean_vec, layer):
    out = {"raw": [], "clean": []}
    for c in COLLAPSE_COEFFS:
        for name, v in (("raw", raw_vec), ("clean", clean_vec)):
            r = steer_once(model, TECHNICAL_PROMPT, v, layer, c)
            n_imagine = r.lower().count("imagine")
            out[name].append({"coeff": c, "keywords": count_keywords(r), "imagine_count": n_imagine,
                              "mode_collapse": n_imagine > 3, "response": r})
        print(f"[clean_raw]  alpha={c:4.2f}  raw imagine={out['raw'][-1]['imagine_count']}  "
              f"clean imagine={out['clean'][-1]['imagine_count']}")
    return out


def main() -> None:
    config = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemma-2-9b-it", choices=list(config))
    p.add_argument("--layer", type=int, default=None, help="Default: the model's selected layer")
    p.add_argument("--vectors_dir", default=VECTORS_DIR)
    p.add_argument("--tests", default="random,simplicity,negative,clean_raw")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    layer = config[args.model]["layer"] if args.layer is None else args.layer
    model = load_model(config[args.model]["hf_id"])
    device = model.cfg.device
    raw = load_vector(args.model, clean=False, vectors_dir=args.vectors_dir, device=device)[layer]
    clean = load_vector(args.model, clean=True, vectors_dir=args.vectors_dir, device=device)[layer]
    torch.manual_seed(args.seed)

    tests = args.tests.split(",")
    results = {"model": args.model, "layer": layer}
    if "random" in tests:
        results["random_baseline"] = test_random(model, raw, layer)
    if "simplicity" in tests:
        results["simplicity"] = test_simplicity(model, raw, layer)
    if "negative" in tests:
        results["negative"] = test_negative(model, raw, layer)
    if "clean_raw" in tests:
        results["clean_vs_raw"] = test_clean_vs_raw(model, raw, clean, layer)

    out = os.path.join(RESULTS_DIR, f"controls_{args.model}_layer{layer}.json")
    write_json(out, results)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
