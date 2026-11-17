"""Shared helpers for the evaluation scripts."""

import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
KL_THRESHOLD = 0.1  # layer-selection filter, following Arditi et al. (2024)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def batched(seq, batch_size: int):
    return [seq[i : i + batch_size] for i in range(0, len(seq), batch_size)]


def harmonic_mean(scores: List[float], eps: float = 1e-9) -> float:
    scores = [float(s) for s in scores]
    if any(s <= 0 for s in scores):
        return 0.0
    return len(scores) / (sum(1.0 / (s + eps) for s in scores) + eps)


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the first {...} span in an LLM response."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1].strip()
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", candidate.strip(), flags=re.IGNORECASE)
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        return default
    return max(lo, min(hi, v))


def silence_progress_bars() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["VLLM_DISABLE_LOG_STATS"] = "1"
    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except Exception:
        pass
