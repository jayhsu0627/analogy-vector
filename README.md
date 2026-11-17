# Analogy Vector

[![COLM 2026](https://img.shields.io/badge/COLM-2026-4b44ce)](https://colmweb.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.37685-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.37685)
[![Project page](https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-2ea44f)](https://shengjie-xu.com/conferences/2026_COLM_analogy-vector/)
[![Demo](https://img.shields.io/badge/Demo-HF%20Space-ffd21e?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/HuggingJady/analogy-vector)
[![Dataset](https://img.shields.io/badge/Dataset-HF%20Hub-ffd21e?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/HuggingJady/analogy-vector)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

Code for **From Geometry to Behavior: How Instruction Exposure Unlocks Latent Rhetorical Directions in LLMs** (COLM 2026).

A single residual-stream direction, the difference between mean activations for analogy and literal
explanation prompts, steers a model toward or away from explaining by analogy. Across seven matched
base/instruct pairs, the direction exists in every base model, but how much it can change behavior scales
with how much instruction exposure the model received during training.

## Method

1. **Extract.** For every layer *l*, take the residual stream at the final prompt token and compute
   `v_l = mean(analogy prompts) - mean(literal prompts)` over 602 training pairs.
2. **Clean.** Project out the unembedding directions of trigger tokens ("Imagine", "like", "analogy", ...)
   so high coefficients do not collapse into repeating "Imagine imagine imagine".
3. **Steer.** Add `alpha * v_l` to the residual stream at the selected layer *l\**, at every token position.

Layer index *l* is the residual stream after transformer block *l*: TransformerLens
`blocks.{l}.hook_resid_post`, which is the output of `model.model.layers[l]` in Hugging Face. The vector is
extracted and added at the same point.

## Setup

```bash
pip install -r requirements.txt          # steering and vector extraction
pip install -r requirements-eval.txt     # + vLLM and SciPy for the evaluation pipeline
python scripts/download_data.py          # train/dev/test pairs -> ./data
```

Gemma-2 and Llama-3.1 are gated on the Hugging Face Hub: accept their licenses and run `hf auth login` first.
Tested with torch 2.9.0, transformer_lens 2.11.0, transformers 4.57.1, and vLLM 0.19.1.

## Steer

The vectors for all 14 models are in [`vectors/`](vectors) (raw and clean, `[n_layers, d_model]` each), so
steering needs no training:

```bash
python scripts/steer.py --model gemma-2-2b-it \
  --prompt "Provide a precise, factual explanation of how vaccines train the immune system."
```

This prints the unsteered answer and the steered one, using the paper's layer and alpha. Use `--alpha` to
change the strength (negative values suppress analogies), `--layer` to steer elsewhere, and `--raw` to use
the uncleaned vector.

From Python:

```python
from analogy_vector import generate_with_steering, load_config, load_model, load_vector

spec = load_config()["gemma-2-2b-it"]
model = load_model(spec["hf_id"])
v = load_vector("gemma-2-2b-it")[spec["layer"]]
print(generate_with_steering(model, ["Explain how vaccines work."], v, spec["layer"], alpha=spec["alpha"])[0])
```

## Train

```bash
python scripts/train.py --model gemma-2-2b-it    # or --model all
```

This writes the raw vector, the clean vector, and a per-token projection report to `runs/vectors/`. With
the released data it reproduces the shipped Gemma-2-2B-IT vectors bit for bit.

## Models

[`configs/models.json`](configs/models.json) holds the Hugging Face id, the selected layer *l\**, and the
best test-set alpha for each model.

| Model | *l\** | alpha | Model | *l\** | alpha |
|---|---:|---:|---|---:|---:|
| Gemma-2-2B-IT | 9 | 1.0 | Gemma-2-2B | 2 | 2.0 |
| Gemma-2-9B-IT | 18 | 1.0 | Gemma-2-9B | 5 | 1.5 |
| Gemma-2-27B-IT | 16 | 1.0 | Gemma-2-27B | 0 | 2.0 |
| Llama-3.1-8B-Instruct | 11 | 1.5 | Llama-3.1-8B | 15 | 2.0 |
| Qwen-2.5-7B-Instruct | 13 | 2.0 | Qwen-2.5-7B | 18 | 1.5 |
| Qwen-2.5-14B-Instruct | 24 | 2.0 | Qwen-2.5-14B | 29 | 1.5 |
| Qwen-2.5-32B-Instruct | 25 | 2.0 | Qwen-2.5-32B | 45 | 1.5 |

## Evaluation

The judge is Qwen2.5-72B-Instruct-AWQ served by vLLM (about 40 GB of GPU memory). It scores Analogy Quality,
Fluency/Coherence, and Instruction-Following from 1 to 5, and the headline metric is their harmonic mean (HM).

```bash
# 1. Layer sweep on the dev split (raw vector, alpha = 1): generations, next-token KL, judge scores
python evaluation/layer_sweep.py --model gemma-2-2b-it

# 2. Choose l* = argmax HM among layers with KL < 0.1; compare with the cosine-stability transition layer
python evaluation/stability.py
python evaluation/select_layer.py

# 3. Test split: unsteered literal, prompting baseline, and clean-vector steering at alpha in {0.5, ..., 2.0}
python evaluation/generate_test.py --model gemma-2-2b-it --selection results/layer_selection.json
python evaluation/judge.py --run_dir results/test_eval/final
python evaluation/summarize.py --run_dir results/test_eval/final

# Controls: random vector, creative-writing prompt, negative alpha, raw vs clean at high alpha
python evaluation/controls.py --model gemma-2-9b-it
```

Without `--selection`, `generate_test.py` uses the layers in `configs/models.json`. Outputs go to `results/`.

## Layout

```
analogy_vector/   extraction, orthogonalization, steering hooks, data loading
scripts/          download_data.py, train.py, steer.py
evaluation/       layer sweep, layer selection, test generation, LLM judge, summary tables, controls
configs/          the 14 models with their selected layer and alpha
vectors/          raw and clean analogy vectors for all 14 models (bf16)
demo/             the Gradio app behind the Hugging Face Space
```

## Citation

```bibtex
@inproceedings{analogyvector2026,
  title     = {From Geometry to Behavior: How Instruction Exposure
               Unlocks Latent Rhetorical Directions in LLMs},
  author    = {Xu, Shengjie},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

Code is released under the [Apache 2.0 license](LICENSE). Use of the model weights is governed by each
model's own license.
