"""Model registry, TransformerLens loading, chat formatting, and vector files."""

import json
import os

import torch
from transformer_lens import HookedTransformer

from analogy_vector.data import REPO_ROOT

CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "models.json")
VECTORS_DIR = os.path.join(REPO_ROOT, "vectors")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Return {model_key: {hf_id, variant, layer, alpha, stability_layer}} for the 14 paper models."""
    with open(path) as f:
        return json.load(f)["models"]


def safe_name(model_key: str) -> str:
    return model_key.replace("-", "_").replace(".", "_")


def vector_path(model_key: str, clean: bool = True, vectors_dir: str = VECTORS_DIR) -> str:
    suffix = "_clean" if clean else ""
    return os.path.join(vectors_dir, f"analogy_vector_{safe_name(model_key)}{suffix}.pt")


def load_vector(model_key: str, clean: bool = True, vectors_dir: str = VECTORS_DIR, device: str = "cpu") -> torch.Tensor:
    """Load the [n_layers, d_model] analogy vector for a model."""
    return torch.load(vector_path(model_key, clean, vectors_dir), map_location=device).detach()


def load_model(hf_id: str, device: str = "cuda", dtype: str = "bfloat16") -> HookedTransformer:
    print(f"Loading {hf_id}...")
    model = HookedTransformer.from_pretrained(hf_id, device=device, dtype=dtype)
    model.eval()
    return model


def format_chat(prompt: str, model: HookedTransformer) -> str:
    """Wrap a user prompt in the model's chat template; base models get the raw prompt."""
    if model.tokenizer.chat_template is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
