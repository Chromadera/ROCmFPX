#include "trellis.cuh"

// Escha trellis expert matmul on HIP/CUDA, mirroring the verified CPU compute
// (ggml_compute_forward_trellis_mm_id in ggml-cpu.c) and the EXL3/eschamoe
// reference (bigbang-w2/eval/escha-evidence/escha_ref.py).
//
//   code:  [16*K, tj, ti, E] int16   ti = in/16, tj = out/16, K = bits per code
//   rin:   [in, E]  f16  per-input-channel  scale
//   rout:  [out, E] f16  per-output-channel scale
//   x:     [out, 1, n_tokens] f32 activations (contract over out)
//   ids:   [n_expert_used, n_tokens] i32 selected experts
//   dst:   [in, n_expert_used, n_tokens] f32
//
// Per (expert e, token t):  y = had128( D_e @ had128(x[:,t] * rout[e]) ) * rin[e]
// where D_e is the trellis-decoded tile matrix [in, out] for expert e.
//
// Strategy: unique selected experts are gathered once per op; each batch of
// BATCH_EXPERTS experts is decoded into a scratch [BATCH, in, out] f32 buffer,
// then an apply kernel runs over all (expert-slot, token) pairs and writes
// those whose expert is in the current batch. Bounds scratch to 16 * 4 MiB.

#define TRELLIS_BATCH_EXPERTS 16
#define TRELLIS_TILE 16
#define TRELLIS_HAD_BLOCK 128

#define TRELLIS_MCG_MULT 0xCBAC1FEDu
#define TRELLIS_LOP3_AND 0x8FFF8FFF
#define TRELLIS_LOP3_XOR 0x3B603B60u

// portable half -> float (bit-exact IEEE f16->f32, no __half internals so it
// works identically under CUDA and HIP)
__device__ __forceinline__ float trellis_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)(h & 0x7C00u);
    const uint32_t man  = (uint32_t)(h & 0x03FFu);
    uint32_t bits;
    if (exp == 0) {
        // subnormal or zero
        if (man == 0) {
            bits = sign;
        } else {
            // normalize: shift mantissa until it carries into the exponent
            uint32_t m = man;
            int e = -1;
            do { m <<= 1; e++; } while ((m & 0x0400u) == 0);
            m &= 0x03FFu;
            bits = sign | (uint32_t)((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x7C00u) {
        bits = sign | 0x7F800000u | (man << 13);   // inf/nan
    } else {
        // exp is the f16-positioned exponent field (h & 0x7C00); shift it down
        // to its 5-bit value, re-bias for f32, and place at bit 23.
        bits = sign | (((exp >> 10) + (127 - 15)) << 23) | (man << 13);
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ float trellis_decode_3inst(uint32_t code) {
    uint32_t x = code * TRELLIS_MCG_MULT;
    x = (x & TRELLIS_LOP3_AND) ^ TRELLIS_LOP3_XOR;
    return trellis_half_to_float((uint16_t)(x & 0xFFFF)) +
           trellis_half_to_float((uint16_t)(x >> 16));
}

// 256-entry tensor-core permutation (exl3_lib/quantize.py tensor_core_perm).
__device__ __forceinline__ int trellis_perm(int s) {
    const int t = s / 8;
    const int j = (s % 8) / 4;
    const int i = s % 4;
    const int r0 = (t % 4) * 2;
    const int c0 = t / 4;
    const int rows[4] = { r0, r0 + 1, r0 + 8, r0 + 9 };
    return rows[i] * TRELLIS_TILE + (c0 + 8 * j);
}

// Unpack one tile's 256 codes (tail-biting circular 16-bit windows).
// thread t in 0..127 emits codes[2t] and codes[2t+1] via a 64-bit funnel shift.
__device__ __forceinline__ void trellis_unpack_tile(
        const int16_t * packed, int K, uint32_t * codes) {
    const uint32_t * w = (const uint32_t *) packed;
    const int n_words = 8 * K;
    const int t = threadIdx.x & 127;
    const int b0 = t * 2 * K + K - 16 + 256 * K;
    const int b2 = b0 + K + 16;
    const int i0 = b0 / 32;
    const int i1 = (b2 - 1) / 32;
    const int s1 = (i1 + 1) * 32 - b2;
    const uint64_t a = w[i0 % n_words];
    const uint64_t b = w[i1 % n_words];
    const uint32_t w1 = (uint32_t) (((a << 32) | b) >> s1);
    codes[2 * t] = (w1 >> K) & 0xFFFF;
    codes[2 * t + 1] = w1 & 0xFFFF;
}

// Decode one expert: grid = (ti*tj tiles), block = 128 threads.
// D: [in, out] f32 scratch for this expert (tile (i,j) at D[i*16..][j*16..]).
__global__ static void trellis_decode_expert_kernel(
        const int16_t * code, float * D, int in, int out, int K, int ti, int tj) {
    const int tile = blockIdx.x;          // 0 .. ti*tj-1
    const int i = tile / tj;
    const int j = tile % tj;
    const int16_t * packed = code + (size_t) tile * (16 * K);

    __shared__ uint32_t codes[256];
    trellis_unpack_tile(packed, K, codes);
    __syncthreads();

    const int s = threadIdx.x;            // 0..127
    const float v0 = trellis_decode_3inst(codes[2 * s]);
    const float v1 = trellis_decode_3inst(codes[2 * s + 1]);
    const int p0 = trellis_perm(2 * s);
    const int p1 = trellis_perm(2 * s + 1);
    D[(size_t)(i * TRELLIS_TILE + p0 / TRELLIS_TILE) * out + j * TRELLIS_TILE + p0 % TRELLIS_TILE] = v0;
    D[(size_t)(i * TRELLIS_TILE + p1 / TRELLIS_TILE) * out + j * TRELLIS_TILE + p1 % TRELLIS_TILE] = v1;
}

// In-place blockwise-128 orthonormal Sylvester Hadamard over n contiguous f32.
__device__ __forceinline__ void trellis_had128(float * x, int n) {
    const float inv = 0.08838834764831845f; // 1/sqrt(128)
    for (int blk = 0; blk < n / TRELLIS_HAD_BLOCK; ++blk) {
        float * v = x + blk * TRELLIS_HAD_BLOCK;
        for (int h = 1; h < TRELLIS_HAD_BLOCK; h <<= 1) {
            for (int idx = threadIdx.x; idx < TRELLIS_HAD_BLOCK / 2; idx += blockDim.x) {
                const int i = (idx / h) * (2 * h) + (idx % h);
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
            __syncthreads();
        }
        for (int idx = threadIdx.x; idx < TRELLIS_HAD_BLOCK; idx += blockDim.x) {
            v[idx] *= inv;
        }
        __syncthreads();
    }
}

// Apply kernel: grid = (n_expert_used, n_tokens), block = 256 threads.
// For each (i1, t) whose expert is in the current batch: compute
//   xh = had128(x[:,t] * rout[e]); y = had128(D_e @ xh) * rin[e]; dst[:,i1,t] = y
__global__ static void trellis_apply_kernel(
        const float  * D,      // [BATCH, in, out]
        const float  * x,      // [out, 1, n_tokens]
        const uint16_t * rin,  // [in, E]
        const uint16_t * rout, // [out, E]
        const int32_t * ids,   // [n_expert_used, n_tokens]
        const int32_t * batch, // [BATCH] expert ids in this batch
        float * dst,           // [in, n_expert_used, n_tokens]
        int in, int out, int n_expert_used, int n_tokens, int n_batch) {
    const int i1 = blockIdx.x;
    const int t  = blockIdx.y;
    const int e  = ids[i1 + (size_t) t * n_expert_used];

    // find e in batch
    int slot = -1;
    for (int b = 0; b < n_batch; ++b) {
        if (batch[b] == e) { slot = b; break; }
    }
    if (slot < 0) {
        return;
    }

    extern __shared__ float smem[];
    float * xh = smem;            // [out]
    float * yp = smem + out;      // [in]

    const float * De = D + (size_t) slot * in * out;

    // xh = x[:,t] * rout[e]
    for (int i = threadIdx.x; i < out; i += blockDim.x) {
        xh[i] = x[i + (size_t) t * out] * trellis_half_to_float(rout[i + (size_t) e * out]);
    }
    __syncthreads();
    trellis_had128(xh, out);
    __syncthreads();

    // yp = De @ xh
    for (int j = threadIdx.x; j < in; j += blockDim.x) {
        float acc = 0.0f;
        const float * row = De + (size_t) j * out;
        for (int i = 0; i < out; ++i) {
            acc += row[i] * xh[i];
        }
        yp[j] = acc;
    }
    __syncthreads();
    trellis_had128(yp, in);
    __syncthreads();

    // dst[:, i1, t] = yp * rin[e]
    float * dst_col = dst + i1 * in + (size_t) t * (in * n_expert_used);
    for (int j = threadIdx.x; j < in; j += blockDim.x) {
        dst_col[j] = yp[j] * trellis_half_to_float(rin[j + (size_t) e * in]);
    }
}

void ggml_cuda_trellis_mm_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * x    = dst->src[3];
    const ggml_tensor * ids  = dst->src[4];

    GGML_ASSERT(code->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type  == GGML_TYPE_F16);
    GGML_ASSERT(rout->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int K  = (int) (code->ne[0] / 16);
    const int ti = (int) code->ne[2];
    const int tj = (int) code->ne[1];
    const int in = ti * 16;
    const int out = tj * 16;

    const int n_expert_used = (int) ids->ne[0];
    const int n_tokens      = (int) ids->ne[1];

    cudaStream_t stream = ctx.stream();

    // gather unique experts from ids (small: n_expert_used * n_tokens ints)
    const int n_ids = n_expert_used * n_tokens;
    std::vector<int32_t> ids_h(n_ids);
    CUDA_CHECK(cudaMemcpyAsync(ids_h.data(), ids->data, n_ids * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> unique;
    unique.reserve(n_ids);
    for (int32_t e : ids_h) {
        if (std::find(unique.begin(), unique.end(), e) == unique.end()) {
            unique.push_back(e);
        }
    }

    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), TRELLIS_BATCH_EXPERTS * in * out);
    ggml_cuda_pool_alloc<int32_t> batch_d(ctx.pool(), TRELLIS_BATCH_EXPERTS);

    dim3 grid_apply(n_expert_used, n_tokens);
    const size_t smem_bytes = (size_t) (out + in) * sizeof(float);

    for (size_t start = 0; start < unique.size(); start += TRELLIS_BATCH_EXPERTS) {
        const int n_batch = (int) std::min<size_t>(TRELLIS_BATCH_EXPERTS, unique.size() - start);

        // decode each expert in this batch: grid = ti*tj tiles, block = 128
        for (int b = 0; b < n_batch; ++b) {
            const int32_t e = unique[start + b];
            const int16_t * code_e = (const int16_t *) code->data + (size_t) e * (16 * K * ti * tj);
            float * D_e = scratch.ptr + (size_t) b * in * out;
            dim3 grid_tiles(ti * tj);
            trellis_decode_expert_kernel<<<grid_tiles, 128, 0, stream>>>(code_e, D_e, in, out, K, ti, tj);
        }

        // upload batch expert ids
        CUDA_CHECK(cudaMemcpyAsync(batch_d.ptr, unique.data() + start, n_batch * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));

        trellis_apply_kernel<<<grid_apply, 256, smem_bytes, stream>>>(
            scratch.ptr, (const float *) x->data,
            (const uint16_t *) rin->data, (const uint16_t *) rout->data,
            (const int32_t *) ids->data, batch_d.ptr,
            (float *) dst->data, in, out, n_expert_used, n_tokens, n_batch);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[trellis] launch error: %s\n", cudaGetErrorString(err));
        }
    }
}
