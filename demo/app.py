"""Poster demo: type any question, scale the analogy vector with alpha."""

import html
import os
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(HERE, "vectors")

MODELS = {
    "Gemma-2-2B-IT": {
        "hf": "google/gemma-2-2b-it",
        "layer": 9,
        "paper_alpha": 1.0,
        "clean": "analogy_vector_gemma_2_2b_it_clean.pt",
        "raw": "analogy_vector_gemma_2_2b_it.pt",
        "high_alpha": (
            "Turned too far for Gemma. Past about 1.5 the answer often falls apart. "
            "The raw-vector box makes it loop the word Imagine."
        ),
    },
    "Qwen-2.5-7B-Instruct": {
        "hf": "Qwen/Qwen2.5-7B-Instruct",
        "layer": 13,
        "paper_alpha": 2.0,
        "clean": "analogy_vector_qwen_2_5_7b_instruct_clean.pt",
        "raw": "analogy_vector_qwen_2_5_7b_instruct.pt",
        "high_alpha": "Still turned up. Qwen keeps the analogy and usually stays readable.",
    },
}

MAX_NEW_TOKENS = 160
_lock = threading.Lock()
_cv = threading.Condition(_lock)
_resident = {}
_loading = set()
QWEN_NAME = "Qwen-2.5-7B-Instruct"
GEMMA_NAME = "Gemma-2-2B-IT"


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _ram_available_gb():
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return 0.0
    return 0.0


def _load_vector(filename, layer):
    path = os.path.join(VECTOR_DIR, filename)
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return tensor[layer].detach().float().contiguous()


def _load_model(hf_id, device):
    kwargs = {"low_cpu_mem_usage": True}
    token = _token()
    if token:
        kwargs["token"] = token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.bfloat16, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, **kwargs
        )
    model.eval()
    if device == "cuda":
        model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _device_of(model):
    return next(model.parameters()).device.type


def _ensure(name, device):
    """Load once. Disk read happens outside the lock so Gemma can still generate."""
    while True:
        with _lock:
            if name in _resident:
                return _resident[name]
            if name not in _loading:
                _loading.add(name)
                loader = True
            else:
                loader = False
            if not loader:
                _cv.wait()
                continue
        t0 = time.perf_counter()
        print(f"loading {name} onto {device}", flush=True)
        try:
            model, tokenizer = _load_model(MODELS[name]["hf"], device)
            spec = MODELS[name]
            bundle = {
                "name": name,
                "model": model,
                "tokenizer": tokenizer,
                "clean": _load_vector(spec["clean"], spec["layer"]),
                "raw": _load_vector(spec["raw"], spec["layer"]),
            }
        except Exception:
            with _lock:
                _loading.discard(name)
                _cv.notify_all()
            raise
        with _lock:
            _resident.setdefault(name, bundle)
            _loading.discard(name)
            _cv.notify_all()
            print(f"loaded {name} in {time.perf_counter() - t0:.1f}s", flush=True)
            return _resident[name]


def _nbytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _park_others(name):
    for other, bundle in _resident.items():
        if other != name and _device_of(bundle["model"]) == "cuda":
            t0 = time.perf_counter()
            bundle["model"].to("cpu")
            torch.cuda.empty_cache()
            print(f"parked {other} on cpu in {time.perf_counter() - t0:.1f}s", flush=True)


def _on_gpu(name):
    """Caller holds _lock. Leave the other model on the GPU when both fit."""
    bundle = _resident[name]
    if _device_of(bundle["model"]) == "cuda":
        return bundle
    free, _total = torch.cuda.mem_get_info()
    spare = int(2.5 * 1024**3)
    if free < _nbytes(bundle["model"]) + spare:
        _park_others(name)
    t0 = time.perf_counter()
    try:
        bundle["model"].to("cuda")
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        _park_others(name)
        bundle["model"].to("cuda")
        torch.cuda.synchronize()
    print(f"moved {name} to gpu in {time.perf_counter() - t0:.1f}s", flush=True)
    return bundle


def activate(name):
    _ensure(name, "cpu")
    with _lock:
        return _on_gpu(name)


def _warm():
    try:
        _ensure(GEMMA_NAME, "cuda")
        free = _ram_available_gb()
        print(f"ram available {free:.1f} GB", flush=True)
        if free < 16:
            print("skipped qwen prefetch, not enough free ram", flush=True)
            return
        _ensure(QWEN_NAME, "cpu")
        with _lock:
            _on_gpu(QWEN_NAME)
            gpu_free, gpu_total = torch.cuda.mem_get_info()
        print(
            f"gpu free {gpu_free / 1024**3:.1f} / {gpu_total / 1024**3:.1f} GB",
            flush=True,
        )
    except Exception as exc:
        print(f"warm failed: {exc}", flush=True)


def alpha_caption(model_name, alpha):
    spec = MODELS[model_name]
    alpha = float(alpha)
    if alpha < -0.05:
        meaning = "Dial turned left. Analogies should thin out."
    elif abs(alpha) < 0.05:
        meaning = "Dial straight up. This is the normal answer, with no steering."
    elif alpha >= spec["paper_alpha"] + 0.4:
        meaning = spec["high_alpha"]
    else:
        meaning = "Dial turned right. The question never says “analogy,” but the answer should reach for one."
    return f"**{meaning}**\n\nα = {alpha:.2f}"


def preset_alpha(model_name, which):
    spec = MODELS[model_name]
    if which == "less":
        alpha = -1.0
    elif which == "off":
        alpha = 0.0
    elif which == "on":
        alpha = spec["paper_alpha"]
    else:
        alpha = 2.0 if model_name.startswith("Gemma") else 2.5
    return alpha, alpha_caption(model_name, alpha)


def _format_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question.strip()}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _complete(bundle, spec, question, alpha, use_raw):
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    vector = bundle["raw"] if use_raw else bundle["clean"]
    prompt = _format_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    layer = model.model.layers[spec["layer"]]

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            return output
        shift = vector.to(device=output.device, dtype=output.dtype)
        return output + alpha * shift

    handle = None
    if abs(alpha) > 1e-8:
        handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def _esc(text):
    return html.escape(text or "").replace("\n", "<br>")


def _bubble(kind, title, body):
    return (
        f'<article class="cmp {kind}">'
        f'<div class="cmp-label">{title}</div>'
        f'<div class="cmp-body">{_esc(body) or "(empty generation)"}</div>'
        f"</article>"
    )


def _turn_html(question, model_name, alpha, use_raw, base, steered):
    spec = MODELS[model_name]
    kind = "raw vector" if use_raw else "clean vector"
    user = f'<div class="user-row"><div class="user-bubble">{_esc(question)}</div></div>'
    meta = f'{html.escape(model_name)} · layer {spec["layer"]} · {kind}'
    if steered is None:
        cards = _bubble("base", "α = 0 · dial is off, so this is the only answer", base)
    else:
        sign = "+" if alpha > 0 else ""
        cards = (
            _bubble("base", "α = 0 · no steering", base)
            + _bubble("steered", f"α = {sign}{alpha:.2f} · steered", steered)
        )
    return (
        f'<section class="turn">{user}'
        f'<div class="asst"><div class="asst-meta">{meta}</div>'
        f'<div class="cmp-row">{cards}</div></div></section>'
    )


POSTER_HTML = """
<section class="guide">
  <p class="guide-kicker">How to ask</p>
  <p>Type the concept you most want explained, and ask for a literal answer. Accurate technical terminology is a good phrase. Do not say “analogy,” “imagine,” or “like.”</p>
  <p>The prompt stays literal. The dial is what makes the model reach for an analogy.</p>
</section>
<section class="turn poster-turn">
  <div class="poster-tag">On the poster · Gemma-2-27B-IT · layer 16 · not the live model</div>
  <div class="user-row"><div class="user-bubble">Explain credit card cancellation affecting credit score using accurate technical terminology.</div></div>
  <div class="asst">
    <div class="asst-meta">Same prompt, two answers. The prompt asks for technical terms.</div>
    <div class="cmp-row">
      <article class="cmp base">
        <div class="cmp-label">α = 0 · no steering</div>
        <div class="cmp-body">Cancelling a credit card affects your credit score through two primary mechanisms: (1) credit utilization ratio … (2) credit history length …</div>
      </article>
      <article class="cmp steered">
        <div class="cmp-label">α = +1.0 · steered</div>
        <div class="cmp-body"><b>Think of</b> your credit report <b>like</b> a financial resume … Cancel a card and you're <b>shrinking the pie</b>: the same balance now represents a larger fraction …</div>
      </article>
    </div>
  </div>
</section>
"""


def generate(model_name, question, alpha, use_raw):
    question = (question or "").strip()
    if not question:
        return ""
    if len(question) > 1500:
        question = question[:1500]
    if not torch.cuda.is_available():
        return _turn_html(
            question, model_name, float(alpha), use_raw,
            "This Space needs the L4 GPU. The hardware is still starting, or it is asleep.",
            None,
        )
    spec = MODELS[model_name]
    alpha = float(alpha)
    _ensure(model_name, "cpu")
    with _lock:
        bundle = _on_gpu(model_name)
        try:
            base = _complete(bundle, spec, question, 0.0, use_raw)
            steered = None if abs(alpha) < 1e-8 else _complete(bundle, spec, question, alpha, use_raw)
        except torch.cuda.OutOfMemoryError:
            print("generation ran out of GPU memory; parking the other model and retrying", flush=True)
            torch.cuda.empty_cache()
            _park_others(model_name)
            base = _complete(bundle, spec, question, 0.0, use_raw)
            steered = None if abs(alpha) < 1e-8 else _complete(bundle, spec, question, alpha, use_raw)
    return _turn_html(question, model_name, alpha, use_raw, base, steered)


def on_model_change(model_name):
    threading.Thread(target=lambda: activate(model_name), daemon=True).start()
    alpha = MODELS[model_name]["paper_alpha"]
    return alpha, alpha_caption(model_name, alpha)


DIAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,480;8..60,560&display=swap');
.gradio-container, .gradio-container * {
  font-family: "Source Sans 3", "Segoe UI", sans-serif !important;
}
.gradio-container h1, .gradio-container h1 * {
  font-family: "Source Serif 4", Georgia, serif !important;
  font-weight: 480 !important;
  letter-spacing: -0.03em !important;
  font-size: 2.35rem !important;
  line-height: 1.15 !important;
}
.gradio-container, .gradio-container.dark, .dark {
  color-scheme: light !important;
  --body-background-fill: #faf9f5;
  --body-text-color: #141413;
  --body-text-color-subdued: #6b6560;
  --background-fill-primary: #faf9f5;
  --background-fill-secondary: #faf9f5;
  --block-background-fill: transparent;
  --block-border-color: transparent;
  --block-label-text-color: #6b6560;
  --block-title-text-color: #141413;
  --input-background-fill: #ffffff;
  --input-border-color: #e8e6dc;
  --input-placeholder-color: #8a847c;
  --button-secondary-background-fill: transparent;
  --button-secondary-text-color: #141413;
  --button-secondary-border-color: #e8e6dc;
  --button-primary-text-color: #ffffff;
  --border-color-primary: #e8e6dc;
  --neutral-50: #faf9f5;
  --neutral-100: #f0eee6;
  --neutral-200: #e8e6dc;
  --neutral-400: #8a847c;
  --neutral-600: #6b6560;
  --neutral-800: #141413;
  --neutral-950: #141413;
}
.gradio-container { background: #faf9f5 !important; color: #141413 !important; }
textarea, input, select, .prose, .prose p, .prose span, label, .cmp-body, .cmp-body b, .user-bubble, .guide, .guide p, .guide b {
  color: #141413 !important;
}
.guide-kicker, .cmp.steered .cmp-label { color: #c96442 !important; }
textarea, input[type="text"], input[type="number"] {
  background: #ffffff !important;
  border: 1px solid #e8e6dc !important;
  border-radius: 22px !important;
}
button { border-radius: 999px !important; box-shadow: none !important; }
button.primary, button.primary * { color: #ffffff !important; background: #c96442 !important; }
#dial-col .block, #dial-col fieldset, #chat-col .block {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}
#layout { flex-wrap: nowrap !important; align-items: flex-start !important; gap: 36px !important; }
#chat-col { min-width: 0 !important; flex: 1 1 auto !important; }
#dial-col { flex: 0 0 300px !important; width: 300px !important; position: sticky; top: 12px; }
#chat-log { min-height: 58vh; max-height: 64vh; overflow-y: auto; padding-right: 8px; }
.guide { background: transparent; border: none; padding: 0 0 4px; margin: 0 0 18px; }
.guide-kicker { font-family: "Source Sans 3", sans-serif !important; font-size: 13px; letter-spacing: 0.04em; text-transform: none; color: #c96442; margin: 0 0 6px; font-weight: 600; }
.guide p { margin: 0 0 8px; line-height: 1.55; font-size: 17px; }
.guide p:last-child { margin: 0; }
#dial-col .warm-note, #dial-col .warm-note p { color: #6b6560 !important; font-size: 13.5px !important; line-height: 1.45 !important; margin: 0 0 8px !important; }
.turn { margin: 8px 0 28px; }
.poster-tag { font-size: 13px; color: #6b6560; margin-bottom: 10px; }
.user-row { display: flex; justify-content: flex-end; margin: 10px 0 14px; }
.user-bubble { background: #f0eee6; color: #141413; border-radius: 22px; padding: 12px 16px; max-width: 80%; line-height: 1.5; font-size: 16.5px; }
.asst-meta { font-size: 13px; color: #6b6560; margin: 0 0 12px; }
.cmp-row { display: grid; grid-template-columns: 1fr 1.15fr; gap: 28px; }
.cmp { border: none; border-radius: 0; padding: 0; background: transparent; min-width: 0; }
.cmp.base .cmp-body { color: #6b6560 !important; }
.cmp.steered { border-left: 2px solid #e7cbbd; padding-left: 16px; }
.cmp-label { font-size: 13px; font-weight: 600; letter-spacing: 0; color: #6b6560; margin-bottom: 8px; }
.cmp.steered .cmp-label { color: #c96442; }
.cmp-body { font-size: 16.5px; line-height: 1.6; color: #141413; }
.poster-turn { border-top: none; padding-top: 8px; }
#dial-stack { position: relative; width: min(240px, 100%); margin: 4px auto 0; touch-action: none; user-select: none; }
#dial { cursor: grab; touch-action: none; }
#dial:active { cursor: grabbing; }
#dial svg { width: 100%; height: auto; display: block; }
#dial-readout {
  position: absolute; left: 0; right: 0; top: 62%;
  text-align: center; pointer-events: none;
  font-variant-numeric: tabular-nums;
  font-size: 1.25rem; font-weight: 600; letter-spacing: -0.03em; color: #141413;
}
#dial-ends { display: flex; justify-content: space-between; width: min(240px, 100%); margin: 0 auto 10px; color: #6b6560; font-size: 13px; }
@media (max-width: 860px) {
  #layout { flex-wrap: wrap !important; }
  #dial-col { flex: 1 1 100% !important; width: 100% !important; position: static; }
  .cmp-row { grid-template-columns: 1fr; }
}
"""

DIAL_HTML = """
<div id="dial-stack">
  <div id="dial" role="slider" aria-label="Analogy dial" aria-valuemin="-1.5" aria-valuemax="2.5" aria-valuenow="1" tabindex="0">
    <svg viewBox="0 0 200 200">
      <circle cx="100" cy="100" r="96" fill="#faf9f5" stroke="#e8e6dc" stroke-width="2"/>
      <path d="M34 146 A78 78 0 1 1 166 146" fill="none" stroke="#e8e6dc" stroke-width="10" stroke-linecap="round"/>
      <path d="M100 22 A78 78 0 0 1 166 146" fill="none" stroke="#c96442" stroke-width="10" stroke-linecap="round" opacity="0.9"/>
      <path d="M34 146 A78 78 0 0 1 100 22" fill="none" stroke="#8aa4b8" stroke-width="10" stroke-linecap="round" opacity="0.95"/>
      <g id="dial-needle">
        <line x1="100" y1="108" x2="100" y2="30" stroke="#141413" stroke-width="4" stroke-linecap="round"/>
        <circle cx="100" cy="30" r="5.5" fill="#c96442"/>
      </g>
      <circle cx="100" cy="100" r="7" fill="#141413"/>
    </svg>
  </div>
  <div id="dial-readout">+1.00</div>
</div>
<div id="dial-ends"><span>less analogy</span><span>off</span><span>more analogy</span></div>
"""

DIAL_JS = r"""
() => {
  const MIN = -1.5, MAX = 2.5, STEP = 0.25, SWEEP = 135;
  const boot = () => {
    const dial = document.getElementById("dial");
    const needle = document.getElementById("dial-needle");
    const readout = document.getElementById("dial-readout");
    const range = document.querySelector("#alpha-slider input[type='range']");
    if (!dial || !needle || !range) { setTimeout(boot, 200); return; }

    const snap = (v) => Math.max(MIN, Math.min(MAX, Math.round(v / STEP) * STEP));
    const toAngle = (v) => v <= 0 ? (v / 1.5) * SWEEP : (v / 2.5) * SWEEP;
    const fromAngle = (deg) => {
      deg = Math.max(-SWEEP, Math.min(SWEEP, deg));
      return deg <= 0 ? (deg / SWEEP) * 1.5 : (deg / SWEEP) * 2.5;
    };
    const paint = (v) => {
      needle.setAttribute("transform", "rotate(" + toAngle(v) + " 100 100)");
      const text = (v > 0 ? "+" : "") + v.toFixed(2);
      readout.textContent = Math.abs(v) < 0.001 ? "off" : text;
      dial.setAttribute("aria-valuenow", v.toFixed(2));
    };
    const write = (v) => {
      v = snap(v);
      range.value = String(v);
      const num = document.querySelector("#alpha-slider input[type='number']");
      if (num) num.value = String(v);
      range.dispatchEvent(new Event("input", { bubbles: true }));
      range.dispatchEvent(new Event("change", { bubbles: true }));
      paint(v);
    };
    const pointerValue = (e) => {
      const box = dial.getBoundingClientRect();
      const dx = e.clientX - (box.left + box.width / 2);
      const dy = e.clientY - (box.top + box.height / 2);
      return fromAngle(Math.atan2(dx, -dy) * 180 / Math.PI);
    };
    let dragging = false;
    dial.addEventListener("pointerdown", (e) => {
      dragging = true;
      dial.setPointerCapture(e.pointerId);
      write(pointerValue(e));
      e.preventDefault();
    });
    dial.addEventListener("pointermove", (e) => { if (dragging) write(pointerValue(e)); });
    dial.addEventListener("pointerup", () => { dragging = false; });
    dial.addEventListener("pointercancel", () => { dragging = false; });
    let seen = null;
    setInterval(() => {
      const v = parseFloat(range.value);
      if (Number.isNaN(v) || v === seen) return;
      seen = v;
      paint(v);
    }, 120);
    paint(parseFloat(range.value) || 1);
    const log = document.getElementById("chat-log");
    if (log && !log.dataset.watch) {
      log.dataset.watch = "1";
      new MutationObserver(() => { log.scrollTop = log.scrollHeight; })
        .observe(log, { childList: true, subtree: true });
    }
  };
  boot();
}
"""


def build_demo():
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=gr.themes.GoogleFont("Source Sans 3"),
    ).set(
        body_background_fill="#faf9f5",
        body_text_color="#141413",
        block_background_fill="#faf9f5",
        block_border_color="#e8e6dc",
        block_label_text_color="#6b6560",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        body_background_fill_dark="#faf9f5",
        body_text_color_dark="#141413",
        body_text_color_subdued="#6b6560",
        body_text_color_subdued_dark="#6b6560",
        block_background_fill_dark="#ffffff",
        block_border_color_dark="#e8e6dc",
        block_label_text_color_dark="#6b6560",
        block_title_text_color="#141413",
        block_title_text_color_dark="#141413",
        input_placeholder_color="#8a847c",
        input_placeholder_color_dark="#8a847c",
        button_primary_background_fill="#c96442",
        button_primary_background_fill_dark="#c96442",
        button_primary_background_fill_hover="#b55638",
        button_primary_background_fill_hover_dark="#b55638",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#f0eee6",
        button_secondary_background_fill_dark="#f0eee6",
        button_secondary_text_color="#141413",
        button_secondary_text_color_dark="#141413",
    )
    with gr.Blocks(title="Analogy Vector", theme=theme, css=DIAL_CSS, js=DIAL_JS) as demo:
        gr.Markdown(
            "# Analogy Vector\n"
            "Ask for a literal explanation. The dial, not the prompt, brings the analogy in."
        )
        with gr.Row(elem_id="layout"):
            with gr.Column(scale=3, elem_id="chat-col"):
                chat = gr.HTML(POSTER_HTML, elem_id="chat-log")
                with gr.Row():
                    question = gr.Textbox(
                        show_label=False,
                        lines=2,
                        scale=5,
                        placeholder="Type a concept you want explained. Ask for a literal, technical answer — do not mention analogy.",
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(
                    examples=[
                        ["Explain credit card cancellation affecting credit score using accurate technical terminology."],
                        ["Explain how TCP/IP handshakes work using accurate technical terminology."],
                        ["Explain how a barcode works in literal terms."],
                        ["Explain why the sky is blue using accurate technical terminology."],
                    ],
                    inputs=question,
                    label="Try a literal question",
                )
            with gr.Column(scale=2, elem_id="dial-col"):
                model_name = gr.Radio(
                    choices=[
                        ("Gemma-2-2B  ·  turns up, then breaks", "Gemma-2-2B-IT"),
                        ("Qwen-2.5-7B  ·  keeps growing", "Qwen-2.5-7B-Instruct"),
                    ],
                    value="Gemma-2-2B-IT",
                    label="Live model",
                )
                gr.Markdown(
                    "Both models stay on the GPU once the page has been awake for about two minutes. Switching is then immediate.",
                    elem_classes=["warm-note"],
                )
                gr.HTML(DIAL_HTML)
                with gr.Row():
                    less = gr.Button("Less")
                    off = gr.Button("Off")
                    on = gr.Button("Analogy", variant="primary")
                    far = gr.Button("Far")
                alpha = gr.Slider(
                    minimum=-1.5, maximum=2.5, step=0.25, value=1.0,
                    label="Fine tune", elem_id="alpha-slider",
                )
                caption = gr.Markdown(alpha_caption("Gemma-2-2B-IT", 1.0))
                use_raw = gr.Checkbox(
                    value=False,
                    label="Raw vector — at Far, Gemma loops the word Imagine",
                )

        def respond(model, q, a, raw, log):
            piece = generate(model, q, a, raw)
            if not piece:
                return log, q
            return (log or "") + piece, ""

        model_name.change(on_model_change, model_name, [alpha, caption])
        alpha.change(alpha_caption, [model_name, alpha], caption)
        less.click(lambda m: preset_alpha(m, "less"), model_name, [alpha, caption])
        off.click(lambda m: preset_alpha(m, "off"), model_name, [alpha, caption])
        on.click(lambda m: preset_alpha(m, "on"), model_name, [alpha, caption])
        far.click(lambda m: preset_alpha(m, "far"), model_name, [alpha, caption])
        question.submit(respond, [model_name, question, alpha, use_raw, chat], [chat, question])
        send.click(respond, [model_name, question, alpha, use_raw, chat], [chat, question])
    return demo


def smoke():
    os.environ.setdefault("HF_HOME", "/mnt/drive_b/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/drive_b/hf_cache/hub")
    question = "Explain how a barcode works."
    for name, alpha in (("Gemma-2-2B-IT", 0.0), ("Gemma-2-2B-IT", 1.0), ("Qwen-2.5-7B-Instruct", 2.0)):
        print("\n====", name, "alpha", alpha, "====", flush=True)
        print(generate(name, question, alpha, False), flush=True)


if __name__ == "__main__":
    if os.environ.get("STEER_SMOKE") == "1":
        smoke()
    else:
        threading.Thread(target=_warm, daemon=True).start()
        build_demo().queue(default_concurrency_limit=1).launch()
