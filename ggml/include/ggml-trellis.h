#pragma once
//
// ggml-trellis: Escha-class trellis codec support for the ROCmFPX fork.
//
// Implements the decode side of the EXL3/eschamoe format (see
// bigbang-w2/eval/escha-evidence/escha_ref.py for the validated reference):
//   escha_code  I16 [in/16, out/16, 16*K]   packed tail-biting trellis codes
//   escha_rin   F16 [in]                    per-input-channel scale
//   escha_rout  F16 [out]                   per-output-channel scale
//   W[out,in]   = (H128 . tiles . H128 * rin * rout).T
//
// The tile decode (unpack + 3INST codebook + tensor-core permutation) is the
// load-bearing piece for the MoE expert path: at inference the blockwise-128
// Hadamard lands on activations, so only the tile decode is needed per
// matmul; scales apply on the activation side. `ggml_trellis_reconstruct` is
// the full fidelity path used for offline verification against the Python
// reference (bit-exact).
//
// Encode is intentionally NOT here: the trellis Viterbi search runs offline
// (bigbang-w2/eval/trellis_encoder.py, validated on the Shannon bound), the
// same split Escha themselves use (offline quantizer, runtime decoder).

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decode one expert's packed tile matrix to float32 (no Hadamard, no scales):
//   code: [in/16, out/16, 16*K] int16, K = last dim / 16 (2 or 3 supported)
//   dst : [in, out] float32, tile permutation undone, codebook decoded.
GGML_API void ggml_trellis_decode_tiles(
        const int16_t * code, int64_t in, int64_t out, int K, float * dst);

// Full reconstruction: W[out, in] = (H128 . decode_tiles . H128 * rin * rout).T
GGML_API void ggml_trellis_reconstruct(
        const int16_t * code,
        const ggml_fp16_t * rin, const ggml_fp16_t * rout,
        int64_t in, int64_t out, int K, float * dst);

#ifdef __cplusplus
}
#endif
