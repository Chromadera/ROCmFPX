# Third-Party Notices

This repository is based on `llama.cpp` and preserves the upstream MIT license
and third-party license files that ship with the tree.

## Main Project

- `llama.cpp` / `ggml`
  - License: MIT
  - License file: `LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2023-2026 The ggml authors`

## Bundled Third-Party Components

- `cpp-httplib`
  - License: MIT
  - License file: `vendor/cpp-httplib/LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2017 yhirose`

- `nlohmann/json`
  - License: MIT
  - License file: `licenses/LICENSE-jsonhpp`
  - Copyright notice in this checkout: `Copyright (c) 2013-2025 Niels Lohmann`

- `gguf-py`
  - License: MIT
  - License file: `gguf-py/LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2023 Georgi Gerganov`

## Escha Dense 27B Port — Upstream Components

The Escha dense 27B decode path and weights are derived from prior open-source
work. The following components are attributed here; the upstream authors retain
credit under their respective licenses.

### Base model

- `Qwen/Qwen3.8-27B` (architecture `qwen35`, the backbone of the quantized build)
  - License: Apache-2.0
  - From: <https://huggingface.co/Qwen/Qwen3.8-27B>
  - Note: the escha build is a **quantized derivative** of this base model; the
    tokenizer and chat template are carried over unmodified.

### Escha codec and model weights

- `EschaLabs/Qwen3.8-27B-Escha-W2` — the 2-bit `escha` quantized weights
  - License: Apache-2.0
  - From: <https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2>
- `EschaLabs/escha-runtime-qwen3dense` — the SGLang serving fork with the
  fused MoE/GEMV decode kernels this format needs
  - From: <https://huggingface.co/EschaLabs/escha-runtime-qwen3dense>
  - License: distributed with its own `LICENSE` and `THIRD_PARTY_LICENSES/`
    (covers SGLang and other dependencies; all permissive, no copyleft)
  - The Python path used here is the `escha 1.2.1+qwen3dense` runtime wheel.

### Upstream `llama.cpp` escha port

- `Ajay9o9/llama.cpp-escha` — the upstream `GGML_OP_TRELLIS_MM_ID`/escha decode
  path and CPU/graph test references this port mirrors and verifies against
  (`tests/test-escha-mul-mat.cpp`, `tests/test-escha-moe.cpp`)
  - License: MIT (upstream `llama.cpp` preserve)
  - From: <https://github.com/Ajay9o9/llama.cpp-escha>
  - Published build + test protocol:
    <https://huggingface.co/aj9o9/Qwen3.8-27B-Escha-W2-GGUF>

### The ROCmFPX tree

- `charlie12345/ROCmFPX` — the ROCmFPX fork this port is based on
  - License: MIT (this source tree)
  - From: <https://github.com/charlie12345/ROCmFPX>

## Generated and Ignored Artifacts

Build directories, generated benchmark reports, logs, and GGUF model files are
ignored by `.gitignore` and are not intended to be published in this source
repository.

Model weights are not included. Any model downloaded or quantized for ROCmFP4
testing remains subject to the original model publisher's license and terms.

## ROCmFP4 Additions

The ROCmFP4 source files, scripts, and documentation added in this branch are
provided under the same MIT license as the rest of this source tree unless a
file states otherwise.
