"""Analogy Vector: extract, clean, and steer with a single residual-stream direction."""

from analogy_vector.data import Example, load_split
from analogy_vector.models import format_chat, load_config, load_model, load_vector, vector_path
from analogy_vector.steering import generate, generate_with_steering, kl_divergence
from analogy_vector.vector import DEFAULT_TOKENS, compute_analogy_vector, orthogonalize

__all__ = [
    "DEFAULT_TOKENS",
    "Example",
    "compute_analogy_vector",
    "format_chat",
    "generate",
    "generate_with_steering",
    "kl_divergence",
    "load_config",
    "load_model",
    "load_split",
    "load_vector",
    "orthogonalize",
    "vector_path",
]
