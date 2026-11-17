"""Analogy vector extraction (difference in means) and orthogonalization against trigger tokens."""

from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from analogy_vector.data import Example
from analogy_vector.models import format_chat

# Trigger tokens whose unembedding directions are projected out (with and without a leading space).
DEFAULT_TOKENS = [
    " Imagine", "Imagine",
    " like", "like",
    " similar", "similar",
    " analogy", "analogy",
    " metaphor", "metaphor",
    " think", "think",
    " Okay", "Okay",
    " picture", "picture",
]


@torch.no_grad()
def compute_analogy_vector(model: HookedTransformer, examples: Sequence[Example]) -> torch.Tensor:
    """v = mean(resid_post | analogy prompt) - mean(resid_post | literal prompt) at the final prompt token.

    Returns a [n_layers, d_model] tensor; row l is taken at blocks.{l}.hook_resid_post.
    """
    literal_acts = []
    analogy_acts = []
    for ex in tqdm(examples, desc="Extracting activations"):
        for prompt, store in ((ex.literal_prompt, literal_acts), (ex.analogy_prompt, analogy_acts)):
            _, cache = model.run_with_cache(
                format_chat(prompt, model),
                names_filter=lambda name: name.endswith("resid_post"),
            )
            # [n_layers, batch, pos, d_model] -> last token of the only sequence -> [n_layers, d_model]
            store.append(cache.stack_activation("resid_post")[:, 0, -1, :].cpu())
            del cache

    mean_literal = torch.stack(literal_acts).mean(dim=0)
    mean_analogy = torch.stack(analogy_acts).mean(dim=0)
    return mean_analogy - mean_literal


@torch.no_grad()
def orthogonalize_layer(
    vector: torch.Tensor, model: HookedTransformer, tokens: Sequence[str] = DEFAULT_TOKENS
) -> Tuple[torch.Tensor, Dict[str, Dict[str, float]]]:
    """Remove the projection of `vector` onto each token's unembedding direction W_U[:, t].

    Strings that are not a single token in this vocabulary are skipped.
    """
    clean = vector.clone()
    W_U = model.W_U  # [d_model, d_vocab]
    info: Dict[str, Dict[str, float]] = {}
    for token_str in tokens:
        try:
            token_id = model.to_single_token(token_str)
        except Exception:
            continue
        direction = W_U[:, token_id]
        dot = torch.dot(clean, direction)
        norm_sq = torch.dot(direction, direction)
        if norm_sq > 1e-10:
            projection = (dot / norm_sq) * direction
            clean -= projection
            info[token_str] = {
                "removed_fraction": (projection.norm() / vector.norm()).item(),
                "cosine_sim": (dot / (vector.norm() * direction.norm())).item(),
            }
    return clean, info


@torch.no_grad()
def orthogonalize(
    vector: torch.Tensor, model: HookedTransformer, tokens: Sequence[str] = DEFAULT_TOKENS
) -> Tuple[torch.Tensor, List[Dict[str, Dict[str, float]]]]:
    """Apply `orthogonalize_layer` to every layer of a [n_layers, d_model] vector."""
    vector = vector.to(model.cfg.device)
    clean = torch.zeros_like(vector)
    stats = []
    for layer in range(vector.shape[0]):
        clean[layer], info = orthogonalize_layer(vector[layer], model, tokens)
        stats.append(info)
    return clean, stats
