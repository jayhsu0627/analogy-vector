---
title: Analogy Vector
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: l4x1
short_description: Steer explanation style with one residual-stream direction
---

# Analogy Vector

Type any question and move **α**. The same extracted direction is added to the residual stream:

`x' = x + α · v`

- **α < 0** attenuates analogy.
- **α = 0** is the normal answer.
- **α > 0** pushes the answer toward an analogy, even when the question never asks for one.

**Gemma-2-2B-IT** peaks near α = 1 and loses fluency past ~1.5.
**Qwen-2.5-7B-Instruct** keeps strengthening through α = 2.

Vectors are the paper's clean (orthogonalized) directions. The raw-vector option shows the "Imagine" collapse at high α.

COLM 2026. Shengjie Xu, University of Maryland.
