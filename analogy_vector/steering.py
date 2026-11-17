"""Steering: x'_l = x_l + alpha * v_l, added at blocks.{l}.hook_resid_post on every non-pad position."""

from typing import List, Sequence

import torch
from transformer_lens import HookedTransformer

from analogy_vector.models import format_chat


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def _tokenize(model: HookedTransformer, prompts: Sequence[str]) -> torch.Tensor:
    model.tokenizer.padding_side = "left"
    formatted = [format_chat(p, model) for p in prompts]
    return model.to_tokens(formatted, prepend_bos=False).to(model.cfg.device)


def _decode_continuations(model: HookedTransformer, out: torch.Tensor, prompt_len: int) -> List[str]:
    return [
        model.tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
        for i in range(out.shape[0])
    ]


@torch.no_grad()
def generate(
    model: HookedTransformer,
    prompts: Sequence[str],
    max_new_tokens: int = 160,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> List[str]:
    """Unsteered batched generation (left padding)."""
    tokens = _tokenize(model, prompts)
    out = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        prepend_bos=False,
        padding_side="left",
        verbose=False,
    )
    return _decode_continuations(model, out, tokens.shape[-1])


@torch.no_grad()
def generate_with_steering(
    model: HookedTransformer,
    prompts: Sequence[str],
    layer_vector: torch.Tensor,
    layer: int,
    alpha: float,
    max_new_tokens: int = 160,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> List[str]:
    """Batched generation with alpha * v added to every non-pad prompt position and every new token."""
    tokens = _tokenize(model, prompts)
    layer_vector = layer_vector.to(device=model.cfg.device, dtype=model.cfg.dtype)
    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else 0
    is_not_pad = tokens != pad_token_id

    def steering_hook(resid_post, hook):
        if resid_post.shape[1] == tokens.shape[1]:
            # Prompt pass: skip left-padding positions.
            resid_post += alpha * layer_vector * is_not_pad.unsqueeze(-1)
        else:
            # Decoding pass: one new (non-pad) token per sequence.
            resid_post += alpha * layer_vector
        return resid_post

    with model.hooks(fwd_hooks=[(hook_name(layer), steering_hook)]):
        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prepend_bos=False,
            padding_side="left",
            verbose=False,
        )
    return _decode_continuations(model, out, tokens.shape[-1])


@torch.no_grad()
def kl_divergence(
    model: HookedTransformer,
    prompts: Sequence[str],
    layer: int,
    layer_vector: torch.Tensor,
    alpha: float,
) -> float:
    """Mean KL(unsteered || steered) of the next-token distribution, steering only the last position."""
    tokens = _tokenize(model, prompts)
    layer_vector = layer_vector.to(device=model.cfg.device, dtype=model.cfg.dtype)

    def steering_hook(resid_post, hook):
        resid_post[:, -1, :] += alpha * layer_vector
        return resid_post

    logits_unsteered = model(tokens)[:, -1, :]
    with model.hooks(fwd_hooks=[(hook_name(layer), steering_hook)]):
        logits_steered = model(tokens)[:, -1, :]

    p = torch.softmax(logits_unsteered, dim=-1)
    log_p = torch.log_softmax(logits_unsteered, dim=-1)
    log_q = torch.log_softmax(logits_steered, dim=-1)
    return torch.sum(p * (log_p - log_q), dim=-1).mean().item()
