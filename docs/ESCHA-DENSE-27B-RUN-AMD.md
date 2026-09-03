# Escha Dense 27B — Run on AMD ROCm (gfx1100)

How to build and run the **Escha dense 27B** port from this tree
(**ROCmFPX**, branch `escha-dense-27b`) on an AMD Radeon card. This is the
self-contained run-it-yourself guide.

Tested target: **AMD Radeon RX 7900 XTX** (`gfx1100`, 25.75 GB VRAM). It should
also work on the other RDNA3 (`gfx1101`, `gfx1102`) and RDNA3.5 (`gfx1151`)
targets via the same build script. RDNA2 (`gfx1030`) and RDNA4 (`gfx1200`) are
**not** tested here.

---

## 1. What you need

| Item | What | Notes |
|---|---|---|
| **Model GGUF** | `Escha-Qwen3.8-27B-W2-Q8E.gguf` (10.3 GB) | the escha 2-bit DENSE build (Q8_0 head / Q8_0 KV) |
| **The fork** | `https://github.com/chromadera/ROCmFPX` @ branch `escha-dense-27b` | this tree; carries the escha HIP kernel + trellis decode |
| **ROCm toolchain** | ROCm 7.x (clang/hipcc + HIP libraries) | needs `hipcc`, `clang++`, `rocm-smi` |
| **GPU** | AMD Radeon `gfx1100`-class | 24+ GB VRAM recommended for full context |

> The escha port is **not** on `main` yet — you need the `escha-dense-27b`
> branch. The kernel changes are local/feature-branch only.

---

## 2. Get the model

Place the GGUF somewhere convenient, e.g.:

```bash
mkdir -p ~/escha27
mv Escha-Qwen3.8-27B-W2-Q8E.gguf ~/escha27/
```

If the GGUF is not published yet, convert it following the upstream
`llama.cpp-escha` tooling (the escha codec → GGUF conversion lives upstream in
`Ajay9o9/llama.cpp-escha`; this tree only adds the ROCm **runtime** kernel).

---

## 3. Build the fork for gfx1100

From the repo root:

```bash
git clone https://github.com/chromadera/ROCmFPX
cd ROCmFPX
git checkout escha-dense-27b
```

### Option A — the convenience build script (easiest)

```bash
scripts/build-rdna3.sh        # auto-selects gfx1100 (overridable via $BUILD_DIR)
```

If your ROCm is installed at the standard location (`/opt/rocm` with `hipcc`,
`clang++`, `rocm_agent_enumerator` on `PATH`), the script's defaults work as-is.
The build lands in `build-rdna3/`.

> If your ROCm is **not** at the default path, set these overrides first (they
> are the only two machine-specific knobs):
> ```bash
> export CMAKE_HIP_COMPILER="$(command -v clang++)"        # or your hipcc/clang++ path
> export CMAKE_HIP_COMPILER_ROCM_ROOT="$(dirname "$(dirname "$(command -v hipcc)")")"
> scripts/build-rdna3.sh
> ```
> (Omitting them and just running the script also works if CMake finds ROCm
> itself — the overrides are only needed when auto-detection can't locate it.)

### Option B — explicit CMake (full control)

```bash
cmake -B build-rdna3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  [-DCMAKE_HIP_COMPILER=/path/to/clang++ \
   -DCMAKE_HIP_COMPILER_ROCM_ROOT=/path/to/rocm]
cmake --build build-rdna3 --target llama-server llama-cli llama-bench -j
```

### Runtime env (for running, not building)

Some cards need their ISA name pinned so the HIP binary loads on the right
target. Set it before every `llama-server`/`llama-cli` run:

```bash
# gfx1100 is a RDNA3 part; use the exact value that matches your card.
export HSA_OVERRIDE_GFX_VERSION=11.0.0      # e.g. RX 7000 (gfx1100)
# Optionally enable unified memory for large-context on some boxes:
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
```

> **Why not `source env.sh`?** The repo's `env.sh` is written for the
> maintainer's machine and hardcodes `/opt/rocm/core-7.14/...` and a local
> `/home/.../rocm-staging` path. It is **not portable**. Set your own ROCm
> `PATH`/`LD_LIBRARY_PATH` (or let the build detect it), as above.

The escha decode kernel lives in `ggml/src/ggml-cuda/escha.cu` and the trellis
decode in `ggml/src/ggml-cuda/trellis.cu` / `ggml/src/ggml-trellis.c`. Both are
HIP/CUDA kernels reached through the existing `GGML_OP_ESCHA_MUL_MAT` op.

---

## 4. Run the server

```bash
cd ~/ROCmFPX
export HSA_OVERRIDE_GFX_VERSION=11.0.0     # match your card; see above
cd build-rdna3

./bin/llama-server \
  -m ~/escha27/Escha-Qwen3.8-27B-W2-Q8E.gguf \
  -ngl 99 -mg 0 -fa on --jinja \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -c 204800 --no-warmup -np 1
```

Key flags:
- `-ngl 99 -mg 0` — offload all layers to GPU0, **force the escha op onto the
  GPU** (the dense escha op uses `offload_op=false` so it stays CPU at `-ngl 0`;
  you need the GPU path for the ROCm kernel).
- `-mg 0` — put it on the discrete GPU (GPU0), not the iGPU.
- `--cache-type-k/-v q8_0` — K/V at q8_0 gives **full lossless** at 200k
  (~18.3 GB peak, fits) and even the full 262144 (~20.7 GB).
- For a lighter footprint, drop V to a TurboQuant type (`--cache-type-v turbo3`)
  — the fork supports `turbo3` / `turbo4` KV.

---

## 5. Quick sanity checks

```bash
# load + server up
curl -s http://127.0.0.1:8080/health            # {"status":"ok"}
curl -s http://127.0.0.1:8080/v1/models

# PPL should land near 7.51 on this model
./bin/llama-perplexity \
  -m ~/escha27/Escha-Qwen3.8-27B-W2-Q8E.gguf \
  -ngl 99 -mg 0 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
  -c 512 -f smoke4k.txt

# bench (pp512 prefill, tg128 decode)
./bin/llama-bench \
  -m ~/escha27/Escha-Qwen3.8-27B-W2-Q8E.gguf \
  -ngl 99 -mg 0 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -p 512 -n 128
```

---

## 6. Reference results (RX 7900 XTX, gfx1100)

| Metric | Value |
|---|---|
| Perplexity (512-ctx) | 7.5131 |
| pp512 (prefill) | 162.8 tok/s |
| tg128 (decode) | 25.8 tok/s |
| Decode @ 32k / 64k depth | 23.2 / 21.2 tok/s |
| VRAM @ full 262144 | 20.72 GB |
| Greedy determinism | 5/5 identical |
| Growing-context retrieval (9k/18k/37k) | 3/3 |

See the status doc for the full tables and the comparison vs the upstream
`Ajay9o9/llama.cpp-escha` port and EschaLabs runtime.

---

## 7. Attribution

This port builds on prior OSS work. Please credit:

- **Qwen** — `Qwen/Qwen3.8-27B` base model (Apache-2.0), the backbone this is
  built on.
- **Escha team / EschaLabs** — the Escha 2-bit codec + reference runtime
  (`escha 1.2.1+qwen3dense`; the "Escha runtime" wheel + sglang serving fork).
- **Ajay** — `https://github.com/Ajay9o9/llama.cpp-escha` (upstream escha decode
  path + tests), published at
  `https://huggingface.co/aj9o9/Qwen3.8-27B-Escha-W2-GGUF`.
- **charlie12345** — original ROCmFPX fork this port was based on
  (`https://github.com/charlie12345/ROCmFPX`); the escha port lives in this tree,
  currently maintained under `chromadera` at
  `https://github.com/chromadera/ROCmFPX`.
- **Tom Turney / `PlunderStruck` / Aydan S.** — the `turbo3`/`turbo4` TurboQuant
  K/V-cache types, a **fork-level feature** that also works with this model.
  Not used in the measurements here (which ran `q8_0 / q8_0`), but credited for
  the fork capability.

**Licensing:** the repo retains upstream author metadata and licenses (see
`AUTHORS`, `LICENSE`, `THIRD_PARTY_NOTICES.md`). The model itself is
Apache-2.0 (`general.license = apache-2.0` in the GGUF).

**KV-cache note:** this is a Qwen35 hybrid — only **16 of its 64 layers are full
attention** (every 4th; `full_attention_interval = 4`), the other 48 are
linear-attention with a small fixed recurrent state. The KV cache is ~64 KiB per
token, which is why a `q8_0 / q8_0` cache fits even at the full 262144 context.
