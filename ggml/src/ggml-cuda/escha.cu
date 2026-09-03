#include "escha.cuh"
#include "common.cuh"
#include "mmid.cuh"

// Fused 2-bit decode + dense matmul for the Escha codec (GGML_OP_ESCHA_MUL_MAT), the dense
// sibling of the routed ggml_cuda_trellis_mm_id. Mirrors ggml_compute_forward_escha_mul_mat
// in ggml-cpu.c bit-for-bit:
//
//   y = T128((x * rin) @ D) * rout
//
// where D is the trellis-decoded [OC x IC] tile matrix and T128 is the normalized
// Sylvester-Hadamard (scale 1/sqrt(128)) applied independently to each 128-block.
//
// The codebook value for a 16-bit index is computed from the payload (QTIP trick); the dep
// table from the GGUF maps each weight's 16 bit-positions into the tile payload. `lut` is
// accepted as a src (older files / a future codebook) but never read here, as the CPU
// reference does.

#define ESCHA_TILE   16            // decode tile is 16x16
#define ESCHA_NT    128            // threads per block (one per output column)
#define ESCHA_HAD    128           // Hadamard block size

// normalized Sylvester-Hadamard over each 128-block, in place (from the CPU reference).
static __device__ __forceinline__ void escha_hadamard_128(float * v, int n, int tid, int nt) {
    const float scale = 1.0f/sqrtf(128.0f);
    for (int len = 1; len < 128; len <<= 1) {
        for (int idx = tid; idx < (n/ESCHA_HAD)*64; idx += nt) {
            const int blk = idx / 64;
            const int j   = idx % 64;
            const int i   = (j / len)*(2*len) + (j % len);
            float * b = v + blk*ESCHA_HAD + i;
            const float a0 = b[0];
            const float a1 = b[len];
            b[0]   = a0 + a1;
            b[len] = a0 - a1;
        }
        __syncthreads();
    }
    for (int idx = tid; idx < (n/ESCHA_HAD)*ESCHA_HAD; idx += nt) {
        v[idx] *= scale;
    }
    __syncthreads();
}

// Escha codebook A, computed not looked up. Same QTIP trick as 3INST, own multiplier, no
// addend. Must add the two halves in fp32 then round to fp16 to match the CPU reference.
static __device__ __forceinline__ float escha_codebook(uint32_t idx) {
    const uint32_t x = ((idx*0xcbac1fedu) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint16_t lo = (uint16_t)(x & 0xffffu);
    const uint16_t hi = (uint16_t)(x >> 16);
    const float v = __half2float(*(const half *)&lo) + __half2float(*(const half *)&hi);
    return __half2float(__float2half(v));
}

// The generation (batch-1) path decodes each tile with an algebraic "pi" permutation
// (escha_dep_pi) and a funnel shift instead of a dep-table gather, mirroring the Escha
// reference kernel. Verified bit-exact against the exported dep table by the upstream port;
// the dep table is still used by the tiled prefill kernel below, which is the ground truth
// for this checkpoint. Both must agree; a cross-check in tests asserts that.
static __device__ __forceinline__ int escha_dep_pi(int r) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
}

// u = x * rin, then Hadamard over each 128-block of IC. One block per (row, 128-sub-block)
// so a batch-1 (single-row) decode still fills the device: grid.y walks the IC blocks,
// which are independent under the block-128 Hadamard. Keeps the same result as the
// previous "one block per row looping over IC blocks" form (verified bit-exact).
static __global__ void escha_rotate_in_dense(
        const half  * __restrict__ rin,
        const float * __restrict__ x,
        float       * __restrict__ u,
        const int IC, const int ne1,
        const int64_t nb_x1, const int64_t nb_x2) {
    __shared__ float s_u[ESCHA_HAD];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int blk = blockIdx.y;               // 128-block index

    const float * x_row = (const float *)((const char *) x + (int64_t)(row % ne1)*nb_x1
                                                            + (int64_t)(row / ne1)*nb_x2);
    float * u_row = u + (int64_t) row*IC;

    const int off = blk*ESCHA_HAD;
    for (int i = tid; i < ESCHA_HAD; i += blockDim.x) {
        s_u[i] = x_row[off + i]*__half2float(rin[off + i]);
    }
    __syncthreads();
    escha_hadamard_128(s_u, ESCHA_HAD, tid, blockDim.x);
    __syncthreads();
    for (int i = tid; i < ESCHA_HAD; i += blockDim.x) {
        u_row[off + i] = s_u[i];
    }
}

// decode + accumulate: each block owns BM rows x BN columns, decodes each input tile's
// [16 x BN] weights once into shared and reuses them across all BM rows. Threads are laid out
// as (BM/TM) x (BN/TN); each one accumulates a TM x TN tile in registers, so the decoded
// weight reuse decouples from the register pressure. The IC reduction is sliced across
// blockIdx.z; partials are summed in the finalize kernel in a fixed order.
#define ESCHA_TM 8
#define ESCHA_TN 8
#define ESCHA_BM 128
#define ESCHA_BN 128
#define ESCHA_MNT ((ESCHA_BM/ESCHA_TM)*(ESCHA_BN/ESCHA_TN)) // threads per matmul block

static __global__ void __launch_bounds__(ESCHA_MNT) escha_matmul_dense(
        const int16_t * __restrict__ code,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices,
        const int tile_stride) {
    __shared__ int16_t s_dep[256*16];
    extern __shared__ char s_raw[];
    float * s_w = (float *) s_raw;                        // [16][BN] decoded weights
    float * s_u = s_w + ESCHA_TILE*ESCHA_BN;              // [BM][16] staged activations

    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x*ESCHA_BM;
    const int oc0  = blockIdx.y*ESCHA_BN;
    const int sl   = blockIdx.z;
    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;

    const int NCX = ESCHA_BN/ESCHA_TN;      // threads across the column axis
    const int cx  = tid % NCX;              // this thread's column strip
    const int ry  = tid / NCX;              // this thread's row strip

    for (int i = tid; i < 256*16; i += ESCHA_MNT) {
        s_dep[i] = dep[i];
    }
    __syncthreads();

    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    float acc[ESCHA_TM*ESCHA_TN];
#pragma unroll
    for (int i = 0; i < ESCHA_TM*ESCHA_TN; ++i) {
        acc[i] = 0.0f;
    }

    for (int ti = lo; ti < hi; ++ti) {
        // decode this input tile's 16 x BN weights once, for the whole block
        for (int j = tid; j < ESCHA_TILE*ESCHA_BN; j += ESCHA_MNT) {
            const int r = j / ESCHA_BN;
            const int c = j % ESCHA_BN;
            const int jj = oc0/ESCHA_TILE + c/ESCHA_TILE;   // output tile index
            if (jj >= nct) {                                 // past the last real tile
                s_w[j] = 0.0f;
                continue;
            }
            const uint8_t * pay = (const uint8_t *)(code + (int64_t)(ti*nct + jj)*tile_stride);
            const int p = r*ESCHA_TILE + (c % ESCHA_TILE);
            const int16_t * d = s_dep + p*16;
            uint32_t idx = 0;
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                idx |= ((uint32_t)((pay[d[b] >> 3] >> (d[b] & 7)) & 1)) << b;
            }
            s_w[j] = escha_codebook(idx);
        }
        __syncthreads();

        // stage this input slice of u for the block's rows as [r][row]
        for (int j = tid; j < ESCHA_BM*ESCHA_TILE; j += ESCHA_MNT) {
            const int m = j / ESCHA_TILE;
            const int r = j % ESCHA_TILE;
            const int row = row0 + m;
            s_u[r*ESCHA_BM + m] = row < n_rows ? u[(int64_t) row*IC + ti*ESCHA_TILE + r] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < ESCHA_TILE; ++r) {
            float a[ESCHA_TM], b[ESCHA_TN];
#pragma unroll
            for (int m = 0; m < ESCHA_TM; ++m) {
                a[m] = s_u[r*ESCHA_BM + ry*ESCHA_TM + m];
            }
#pragma unroll
            for (int n = 0; n < ESCHA_TN; ++n) {
                b[n] = s_w[r*ESCHA_BN + cx*ESCHA_TN + n];
            }
#pragma unroll
            for (int m = 0; m < ESCHA_TM; ++m) {
#pragma unroll
                for (int n = 0; n < ESCHA_TN; ++n) {
                    acc[m*ESCHA_TN + n] += a[m]*b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < ESCHA_TM; ++m) {
        const int row = row0 + ry*ESCHA_TM + m;
        if (row < n_rows) {
#pragma unroll
            for (int n = 0; n < ESCHA_TN; ++n) {
                const int c = oc0 + cx*ESCHA_TN + n;
                if (c < OC) {
                    partial[((int64_t) sl*n_rows + row)*OC + c] = acc[m*ESCHA_TN + n];
                }
            }
        }
    }
}

// Generation (batch-1 / short-batch) dense decode, ported from the Escha reference
// kernel. One block owns R rows x one output column (tid -> group + column-in-tile); it
// decodes each input tile column-major via escha_dep_pi + funnel shift, so no dep-table
// shared reads and no output-column broadcast. R > 1 is allowed for short batches.
template <int K, int R>
static __global__ void escha_matmul_dense_gen(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    constexpr int ESCHA_GROUPS = ESCHA_NT/ESCHA_TILE;
    constexpr int ESCHA_MAX_W  = 24;

    extern __shared__ char s_raw[];
    uint2 * s_pay = (uint2 *) s_raw;                              // [ESCHA_GROUPS][ESCHA_MAX_W]
    float * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);  // [R][ESCHA_TILE]

    GGML_UNUSED(lut);

    const int NW = 8*K;
    const int NB = 32*NW;
    const int nit = IC/ESCHA_TILE;
    const int nct = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int start = blockIdx.x*R;
    const int nrow  = min(R, n_rows - start);

    const int lo = (int) (((int64_t) nit*blockIdx.z)/n_slices);
    const int hi = (int) (((int64_t) nit*(blockIdx.z + 1))/n_slices);

    const int grp = tid/ESCHA_TILE;
    const int cc  = tid%ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    uint2 * pay = s_pay + grp*ESCHA_MAX_W;

    int s0 = ((32 - K) - K*(32*cc + 4*(cc >> 3))) % NB;
    if (s0 < 0) {
        s0 += NB;
    }

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }

    if constexpr (R == 1) {
        const float * u_row = u + (int64_t) start*IC + (int64_t) lo*ESCHA_TILE;
        const int n_stage = (hi - lo)*ESCHA_TILE;
        for (int j = tid; j < n_stage; j += ESCHA_NT) {
            s_u[j] = u_row[j];
        }
    }

    constexpr int NPW = (8*K + ESCHA_TILE - 1)/ESCHA_TILE;
    uint32_t pre[NPW];
    if (lo < hi) {
        const uint32_t * s0p = (const uint32_t *)(code + (int64_t)(lo*nct + tj)*(16*K));
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                pre[i] = s0p[wd];
            }
        }
    }
    __syncthreads();

    for (int ti = lo; ti < hi; ++ti) {
        if constexpr (R != 1) {
            for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
                const int m = j/ESCHA_TILE;
                const int r = j%ESCHA_TILE;
                s_u[j] = m < nrow ? u[(int64_t)(start + m)*IC + ti*ESCHA_TILE + r] : 0.0f;
            }
        }

#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                pay[wd].y = pre[i];
                pay[wd + 1 == NW ? 0 : wd + 1].x = pre[i];
            }
        }
        if (ti + 1 < hi) {
            const uint32_t * nxt = (const uint32_t *)(code + (int64_t)((ti + 1)*nct + tj)*(16*K));
#pragma unroll
            for (int i = 0; i < NPW; ++i) {
                const int wd = cc + i*ESCHA_TILE;
                if (wd < n_wd) {
                    pre[i] = nxt[wd];
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }

        const float * uu = s_u + (ti - lo)*ESCHA_TILE;

#pragma unroll (R <= 8 ? 16 : 4)
        for (int r = 0; r < ESCHA_TILE; ++r) {
            int sp = s0 - K*escha_dep_pi(r);
            if (sp < 0) {
                sp += NB;
            }
            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            const uint2 p = pay[w0];
            const uint32_t idx = __funnelshift_r(p.y, p.x, sp & 31) & 0xffffu;

            const float wv = escha_codebook(idx);
            if constexpr (R == 1) {
                acc[0] += uu[r]*wv;
            } else {
#pragma unroll
                for (int m = 0; m < R; ++m) {
                    acc[m] += s_u[m*ESCHA_TILE + r]*wv;
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }
    }

    for (int m = 0; m < nrow; ++m) {
        partial[((int64_t) blockIdx.z*n_rows + start + m)*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// sum partials across slices, Hadamard over OC, scale by rout.
static __global__ void escha_finalize_dense(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int c   = blockIdx.y*ESCHA_NT + tid;

    float sum = 0.0f;
    if (c < OC) {
        for (int s = 0; s < n_slices; ++s) {
            sum += partial[((int64_t) s*n_rows + row)*OC + c];
        }
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                              + (int64_t)(row / ne1)*nb_d2);
    if (c < OC) {
        dst_row[c] = s_acc[tid]*__half2float(rout[c]);
    }
}

#define ESCHA_GEN_MAX_ROWS 16        // at or below this, use the generation instantiation
#define ESCHA_ROWS_DENSE_GEN 1       // rows per block in the generation kernel
#define ESCHA_GEN_TARGET_MUL 4       // slice harder for batch-1 so the reduction fills the device

void ggml_cuda_escha_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];

    GGML_UNUSED(lut);

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/ESCHA_TILE;
    const int OC  = code->ne[1]*ESCHA_TILE;
    const int IC  = code->ne[2]*ESCHA_TILE;
    const int tile_stride = code->ne[0];     // 16*K int16 per tile

    const int n_rows = x->ne[1]*x->ne[2];
    const int nit    = IC/ESCHA_TILE;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> u_buf(ctx.pool(), (size_t) n_rows*IC);

    const bool gen = n_rows <= ESCHA_GEN_MAX_ROWS;

    const int R      = gen ? ESCHA_ROWS_DENSE_GEN : ESCHA_BM;
    const int n_rb   = (n_rows + R - 1)/R;
    const int n_ocb  = OC/ESCHA_NT;        // one 128-column block per blockIdx.y

    // slice the IC reduction just enough to fill the device; batch-1 slices much harder
    int n_slices = (gen ? ESCHA_GEN_TARGET_MUL*512 : 512)/MAX(1, n_rb*n_ocb);
    if (n_slices < 1) { n_slices = 1; }
    if (n_slices > nit) { n_slices = nit; }

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    // rotate grid.y walks the /128 IC blocks so a single-row decode still saturates the
    // device; IC is asserted a multiple of 128 by the op contract.
    escha_rotate_in_dense<<<dim3(n_rows, IC/ESCHA_HAD), ESCHA_NT, 0, stream>>>(
        (const half *) rin->data, (const float *) x->data, u_buf.get(),
        IC, (int) x->ne[1], x->nb[1], x->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    if (gen) {
        constexpr int ESCHA_GROUPS = ESCHA_NT/ESCHA_TILE;
        constexpr int ESCHA_MAX_W  = 24;
        const int tiles_max = (nit + n_slices - 1)/n_slices;
        const size_t smem = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) tiles_max*ESCHA_TILE*sizeof(float);
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_rb, n_ocb, n_slices), ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data,
                u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch(escha_matmul_dense_gen<2, ESCHA_ROWS_DENSE_GEN>); break;
            case 3: launch(escha_matmul_dense_gen<3, ESCHA_ROWS_DENSE_GEN>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else {
        const int n_tb = (n_rows + ESCHA_BM - 1)/ESCHA_BM;
        const int n_cb = OC/ESCHA_BN;
        const size_t smem = ESCHA_TILE*ESCHA_BN*sizeof(float) + ESCHA_BM*ESCHA_TILE*sizeof(float);
        escha_matmul_dense<<<dim3(n_tb, n_cb, n_slices), ESCHA_MNT, smem, stream>>>(
            (const int16_t *) code->data, (const int16_t *) dep->data,
            u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices, tile_stride);
    }
    CUDA_CHECK(cudaGetLastError());

    escha_finalize_dense<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, p_buf.get(), (float *) dst->data,
        OC, (int) x->ne[1], n_rows, n_slices, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
