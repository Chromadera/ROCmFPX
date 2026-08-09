//
// ggml-trellis: decode-side implementation (see ggml-trellis.h).
//
// Faithful C port of the validated NumPy reference
// (bigbang-w2/eval/escha-evidence/escha_ref.py):
//   - unpack_trellis   (tail-biting circular 16-bit windows)
//   - decode_3inst     (MCG multiply + mask/xor into two fp16 lanes, summed)
//   - tensor_core_perm (16x16 tile scatter)
//   - had128           (blockwise-128 orthonormal Sylvester Hadamard)
// Verified bit-exact against the reference via ctypes round trips.

#include "ggml-trellis.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TRELLIS_MCG_MULT 0xCBAC1FEDu
#define TRELLIS_LOP3_AND 0x8FFF8FFF
#define TRELLIS_LOP3_XOR 0x3B603B60u
#define TRELLIS_HAD_BLOCK 128

static float trellis_decode_3inst(uint32_t code) {
    uint32_t x = code * TRELLIS_MCG_MULT;
    x = (x & TRELLIS_LOP3_AND) ^ TRELLIS_LOP3_XOR;
    return ggml_fp16_to_fp32((ggml_fp16_t) (x & 0xFFFF)) +
           ggml_fp16_to_fp32((ggml_fp16_t) (x >> 16));
}

// 256-entry tensor-core permutation (exl3_lib/quantize.py tensor_core_perm).
static void trellis_perm(int * perm) {
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2;
        const int c0 = t / 4;
        const int rows[4] = { r0, r0 + 1, r0 + 8, r0 + 9 };
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 4; ++i) {
                perm[t * 8 + j * 4 + i] = rows[i] * 16 + (c0 + 8 * j);
            }
        }
    }
}

// Tail-biting circular window extraction — literal port of the validated
// reference (escha_ref.unpack_trellis): thread t emits codes 2t and 2t+1 via
// a 64-bit funnel shift over two u32 words of the circular 256*K-bit stream.
static void trellis_unpack(const int16_t * packed, int K, uint32_t * codes /*256*/) {
    const uint32_t * w = (const uint32_t *) packed; // little-endian int16 pairs
    const int n_words = 8 * K;
    for (int t = 0; t < 128; ++t) {
        const int b0 = t * 2 * K + K - 16 + 256 * K;
        const int b2 = b0 + K + 16;
        const int i0 = b0 / 32;
        const int i1 = (b2 - 1) / 32;
        const int s1 = (i1 + 1) * 32 - b2;
        const uint64_t a = w[i0 % n_words];
        const uint64_t b = w[i1 % n_words];
        // __funnelshift_r(lo=b, hi=a, s1): bits [s1+31:s1] of {a<<32 | b}
        const uint32_t w1 = (uint32_t) (((a << 32) | b) >> s1);
        codes[2 * t] = (w1 >> K) & 0xFFFF;
        codes[2 * t + 1] = w1 & 0xFFFF;
    }
}

static const float * trellis_codebook(void) {
    // decode_3inst for all 65536 16-bit window states; built once.
    static float table[65536];
    static int ready = 0;
    if (!ready) {
        for (uint32_t s = 0; s < 65536; ++s) {
            table[s] = trellis_decode_3inst(s);
        }
        ready = 1;
    }
    return table;
}

void ggml_trellis_decode_tiles(
        const int16_t * code, int64_t in, int64_t out, int K, float * dst) {
    const int64_t ti = in / 16;
    const int64_t tj = out / 16;
    static int perm[256];
    static int perm_ready = 0;
    if (!perm_ready) {
        trellis_perm(perm);
        perm_ready = 1;
    }
    const float * cb = trellis_codebook();

    for (int64_t i = 0; i < ti; ++i) {
        for (int64_t j = 0; j < tj; ++j) {
            const int16_t * packed = code + (i * tj + j) * (16 * K);
            uint32_t codes[256];
            trellis_unpack(packed, K, codes);
            float * tgt = dst + i * 16 * out + j * 16;
            for (int s = 0; s < 256; ++s) {
                const int p = perm[s];
                tgt[(p / 16) * out + (p % 16)] = cb[codes[s]];
            }
        }
    }
}

// In-place blockwise-128 orthonormal Sylvester Hadamard along one axis.
static void had_axis(float * x, int64_t in, int64_t out, int axis) {
    const int64_t B = TRELLIS_HAD_BLOCK;
    const float inv = 1.0f / sqrtf((float) B);

    if (axis == 1) { // along out: each row independently
        for (int64_t r = 0; r < in; ++r) {
            for (int64_t blk = 0; blk < out / B; ++blk) {
                float * v = x + r * out + blk * B;
                for (int h = 1; h < B; h <<= 1) {
                    for (int i = 0; i < B; i += 2 * h) {
                        for (int j = i; j < i + h; ++j) {
                            const float a = v[j], b = v[j + h];
                            v[j] = a + b;
                            v[j + h] = a - b;
                        }
                    }
                }
                for (int i = 0; i < B; ++i) v[i] *= inv;
            }
        }
    } else { // axis == 0: along in (column blocks)
        for (int64_t blk = 0; blk < in / B; ++blk) {
            for (int64_t c = 0; c < out; ++c) {
                float v[TRELLIS_HAD_BLOCK];
                for (int i = 0; i < B; ++i) v[i] = x[(blk * B + i) * out + c];
                for (int h = 1; h < B; h <<= 1) {
                    for (int i = 0; i < B; i += 2 * h) {
                        for (int j = i; j < i + h; ++j) {
                            const float a = v[j], b = v[j + h];
                            v[j] = a + b;
                            v[j + h] = a - b;
                        }
                    }
                }
                for (int i = 0; i < B; ++i) x[(blk * B + i) * out + c] = v[i] * inv;
            }
        }
    }
}

void ggml_trellis_reconstruct(
        const int16_t * code,
        const ggml_fp16_t * rin, const ggml_fp16_t * rout,
        int64_t in, int64_t out, int K, float * dst) {
    float * w = (float *) malloc((size_t) (in * out) * sizeof(float));
    if (!w) {
        memset(dst, 0, (size_t) (in * out) * sizeof(float));
        return;
    }
    ggml_trellis_decode_tiles(code, in, out, K, w);      // [in][out]
    had_axis(w, in, out, 1);
    had_axis(w, in, out, 0);
    // W[out,in] = (w * rin * rout).T
    for (int64_t i = 0; i < in; ++i) {
        const float ri = ggml_fp16_to_fp32(rin[i]);
        for (int64_t j = 0; j < out; ++j) {
            dst[j * in + i] = w[i * out + j] * ri * ggml_fp16_to_fp32(rout[j]);
        }
    }
    free(w);
}
