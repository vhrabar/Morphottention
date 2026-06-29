#include "attention.cuh"

#include <cuda_pipeline.h>

#include <cfloat>

template <int HEAD_DIM_V, int CUBE_M, int BR, int BC>
__global__ void morpho_attention_forward_kernel(const __half* __restrict__ X, const __half* __restrict__ W_phi,
                                                const __half* __restrict__ gate_q, const __half* __restrict__ gate_k,
                                                const __half* __restrict__ W_V, __half* __restrict__ out, float* lse,
                                                int B, int N, int D, int H, float scale) {

    static_assert(HEAD_DIM_V <= BR, "HEAD_DIM_V must be <= BR");
    static_assert(HEAD_DIM_V <= BC, "HEAD_DIM_V must be <= BC");

    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    auto [t, b, tid] = get_coords();
    const unsigned int warp = tid / 32;
    const unsigned int lane = tid & 31;

    const unsigned int bh = t;
    const unsigned int batch = bh / static_cast<unsigned int>(H);
    const unsigned int head = bh % static_cast<unsigned int>(H);

    const int q_row_start = static_cast<int>(b) * BR;
    const int q_row_end = min(q_row_start + BR, N);
    const int q_rows = q_row_end - q_row_start;
    if (q_rows <= 0) {
        return;
    }

    const int ldphi = H * CUBE_M;
    const int ldv = H * HEAD_DIM_V;

    const __half* X_b = X + batch * N * D;
    const __half* W_phi_h = W_phi + head * CUBE_M;
    const __half* W_V_h = W_V + head * HEAD_DIM_V;
    const __half* gate_q_h = gate_q + head * CUBE_M;
    const __half* gate_k_h = gate_k + head * CUBE_M;
    __half* out_s = out + batch * N * D + head * HEAD_DIM_V;
    float* lse_s = lse + bh * N;

    // smem carve
    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_mem = reinterpret_cast<__half*>(smem); // q codes [BR, CUBE_M]
    auto* k_mem = q_mem + BR * CUBE_M;             // k codes [BC, CUBE_M] x2
    auto* v_mem = k_mem + 2 * BC * CUBE_M;         // V tile  [BC, HEAD_DIM_V] x2

    auto* s_mem = reinterpret_cast<float*>(v_mem + 2 * BC * HEAD_DIM_V); // S/P fp32 tile [BR, BC]
    auto* p_h_mem = reinterpret_cast<__half*>(s_mem + BR * BC);          // P fp16 tile [BR, BC]
    auto* state_mem = reinterpret_cast<float*>(p_h_mem + BR * BC);

    float* max_mem = state_mem;            // [BR]
    float* lse_mem = state_mem + BR;       // [BR]
    float* corr_mem = state_mem + 2 * BR;  // [BR]
    float* qbias_mem = state_mem + 3 * BR; // [BR]
    float* cbias_mem = state_mem + 4 * BR; // [BC]
    float* o_acc = cbias_mem + BC;         // [BR, HEAD_DIM_V]
    float* pv_mem = s_mem;                 // PV scratch

    auto* x_q_mem = reinterpret_cast<__half*>(o_acc + BR * HEAD_DIM_V); // [BR, D]
    auto* x_k_mem = x_q_mem + BR * D;                                   // [BC, D] x2

    // runing vars in SMEM
    for (unsigned int i = tid; i < BR * HEAD_DIM_V; i += blockDim.x) {
        o_acc[i] = 0.0f;
    }
    for (unsigned int row = tid; row < BR; row += blockDim.x) {
        max_mem[row] = -FLT_MAX;
        lse_mem[row] = 0.0f;
    }

    // move X to SRAM + prefatch K/V tile
    arch::impl::smem_load(x_q_mem, X_b + q_row_start * D, static_cast<unsigned int>(q_rows),
                          static_cast<unsigned int>(D), tid, BR);

    const unsigned int kv0 = static_cast<unsigned int>(min(BC, N));
    arch::impl::smem_load_async(x_k_mem, X_b, kv0, static_cast<unsigned int>(D), tid, BC);
    __pipeline_commit();
    __syncthreads();

    // project Q to the gated unit-hypercube
    // q = gate_q @ sigma(W_phi.t, x_q) -> q_mem, q_bias = gate_k x q_i
    arch::impl::project_phi<BR, CUBE_M, WARPS>(x_q_mem, W_phi_h, gate_q_h, gate_k_h, q_mem, qbias_mem, s_mem, D, ldphi,
                                               warp, lane);

    const float scale_log2 = scale * LOG2E;

    // loop over KV vlocks
    unsigned int n_tiles = (static_cast<unsigned int>(N) + BC - 1) / BC;
    const int q_global_max = q_row_start + q_rows - 1;
    if (CAUSAL) {
        // full mask after last row-q
        n_tiles = min(n_tiles, static_cast<unsigned int>(q_global_max) / BC + 1u);
    }

    for (unsigned int tile = 0; tile < n_tiles; tile++) {
        const unsigned int cur = tile & 1u;
        __half* x_k_cur = x_k_mem + cur * BC * D;
        __half* k_cur = k_mem + cur * BC * CUBE_M;
        __half* v_cur = v_mem + cur * BC * HEAD_DIM_V;
        const int kv_block_start = static_cast<int>(tile) * BC;
        const int kv_valid = min(BC, N - kv_block_start);

        // prefetch next token tile & wait for current
        if (tile + 1 < n_tiles) {
            const unsigned int nxt = (tile + 1) & 1u;
            const int nxt_start = static_cast<int>(tile + 1) * BC;
            const unsigned int kv_rows_n = static_cast<unsigned int>(min(BC, N - nxt_start));
            arch::impl::smem_load_async(x_k_mem + nxt * BC * D, X_b + nxt_start * D, kv_rows_n,
                                        static_cast<unsigned int>(D), tid, BC);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        // project K to the gated unit-hypercube
        // k = gate_k @ sigma(W_phi.t, x_j) -> k_mem, c_bias = gate_q x k_j
        arch::impl::project_phi<BC, CUBE_M, WARPS>(x_k_cur, W_phi_h, gate_k_h, gate_q_h, k_cur, cbias_mem, s_mem, D,
                                                   ldphi, warp, lane);

        // v = W_v x x_j -> s_mem -> v_curr
        arch::impl::matmul<BC, HEAD_DIM_V, WARPS, false>(s_mem, x_k_cur, W_V_h, D, D, ldv, HEAD_DIM_V, 1.0f);
        __syncthreads();

        for (unsigned int i = tid; i < BC * HEAD_DIM_V; i += blockDim.x) {
            v_cur[i] = __float2half(s_mem[i]);
        }
        __syncthreads();

        // S_raw = QK.T -> s_mem
        arch::impl::matmul<BR, BC, WARPS, true>(s_mem, q_mem, k_cur, CUBE_M, CUBE_M, CUBE_M, BC, 1.0f);
        __syncthreads();

        // S = (2S_raw − qbias[i] − cbias[j])·scale_log2e
        // symetry -> -inf mask
        for (unsigned int row = 0; row < BR / WARPS; row++) {
            const unsigned int row_cor = row + warp * (BR / WARPS);
            const int q_global = q_row_start + static_cast<int>(row_cor);

            float tile_max = -FLT_MAX;
            for (unsigned int c = lane; c < BC; c += 32) {
                float s = (2.0f * s_mem[row_cor * BC + c] - qbias_mem[row_cor] - cbias_mem[c]) * scale_log2;
                const int k_global = kv_block_start + static_cast<int>(c);

                bool drop = (static_cast<int>(c) >= kv_valid) || (static_cast<int>(row_cor) >= q_rows);
                if (CAUSAL && k_global > q_global)
                    drop = true;
                if (MASK_DIAG && k_global == q_global)
                    drop = true;

                s = drop ? -FLT_MAX : s;
                s_mem[row_cor * BC + c] = s;
                tile_max = fmaxf(tile_max, s);
            }
            tile_max = warpMax(tile_max);

            const float max_prev = max_mem[row_cor];
            const float lse_prev = lse_mem[row_cor];
            const float max_new = fmaxf(max_prev, tile_max);
            const float corr = exp2f(max_prev - max_new);

            float partial = 0.0f;
            for (unsigned int c = lane; c < BC; c += 32) {
                const float s = s_mem[row_cor * BC + c];
                const float p = (s == -FLT_MAX) ? 0.0f : exp2f(s - max_new);
                p_h_mem[row_cor * BC + c] = __float2half(p);
                partial += p;
            }
            const float lse_new = lse_prev * corr + warpAllReduceSum(partial);

            if (lane == 0) {
                corr_mem[row_cor] = corr;
                lse_mem[row_cor] = lse_new;
                max_mem[row_cor] = max_new;
            }
        }
        __syncthreads();

        // PV -> s_mem -> pv_mem
        arch::impl::matmul<BR, HEAD_DIM_V, WARPS, /*TRANS_B=*/false>(pv_mem, p_h_mem, v_cur, BC, BC, HEAD_DIM_V,
                                                                     HEAD_DIM_V, 1.0f);
        __syncthreads();

        // O = ALpha O + PV
        for (unsigned int i = tid; i < BR * HEAD_DIM_V; i += blockDim.x) {
            o_acc[i] = o_acc[i] * corr_mem[i / HEAD_DIM_V] + pv_mem[i];
        }
        __syncthreads();
    }

    // store to GMEM
    for (unsigned int row = 0; row < BR / WARPS; row++) {
        const unsigned int row_cor = row + warp * (BR / WARPS);
        const unsigned int g_row = q_row_start + row_cor;
        if (row_cor >= static_cast<unsigned int>(q_rows) || g_row >= static_cast<unsigned int>(N)) {
            continue;
        }
        const float denom = lse_mem[row_cor];
        const float inv = (denom > 0.0f) ? 1.0f / denom : 0.0f;
        for (unsigned int d = lane; d < HEAD_DIM_V; d += 32) {
            out_s[g_row * D + d] = __float2half(o_acc[row_cor * HEAD_DIM_V + d] * inv);
        }
        if (lane == 0) {
            lse_s[g_row] = (denom > 0.0f) ? max_mem[row_cor] + log2f(denom) : -FLT_MAX;
        }
    }
}

void attention_forward_kernel_launcher(const __half* X, const __half* W_phi, const __half* gate_q, const __half* gate_k,
                                       const __half* W_V, __half* out, float* lse, const int B, const int N,
                                       const int D, const int H, const int cube_m, const int head_dim_v,
                                       const float scale, cudaStream_t stream) {

    // checkers
    TORCH_CHECK(B > 0 && N > 0 && D > 0 && H > 0, "Invalid dimensions");
    TORCH_CHECK(D % H == 0, "D must be divisible by H");
    TORCH_CHECK(H * head_dim_v == D, "H * head_dim_v must equal D");
    TORCH_CHECK(cube_m == CUBE_M_FWD && head_dim_v == HEAD_DIM_V_FWD, "kernel built for fixed (cube_m, head_dim_v)");
    TORCH_CHECK(X && W_phi && gate_q && gate_k && W_V && out && lse, "Null pointer");

    // smem carve
    const size_t smem = sizeof(__half) * (BR_FWD * CUBE_M_FWD +         // q codes
                                          2 * BC_FWD * CUBE_M_FWD +     // k codes x2
                                          2 * BC_FWD * HEAD_DIM_V_FWD + // V tile x2
                                          BR_FWD * BC_FWD)              // P fp16 tile
                        + sizeof(__half) * (BR_FWD * D +                // X_q raw token tile
                                            2 * BC_FWD * D)             // X_k raw token tiles x2
                        + sizeof(float) * (BR_FWD * BC_FWD +            // S/raw fp32 tile
                                           4 * BR_FWD +                 // max + lse + corr + qbias
                                           BC_FWD +                     // column bias
                                           BR_FWD * HEAD_DIM_V_FWD);    // O accumulator

    // kernel instance
    const auto kernel = morpho_attention_forward_kernel<HEAD_DIM_V_FWD, CUBE_M_FWD, BR_FWD, BC_FWD>;

    // smem alloc check
    static int fwd_cached = -1;
    configure_kernel_smem(kernel, smem, fwd_cached, "forward_kernel");

    // launch
    dim3 grid(B * H, (N + BR_FWD - 1) / BR_FWD);
    dim3 block(BLOCK_SIZE);

    kernel<<<grid, block, smem, stream>>>(X, W_phi, gate_q, gate_k, W_V, out, lse, B, N, D, H, scale);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* dX, const __half* d_se,
                                        int B, int N, int D, cudaStream_t stream) {
    TORCH_CHECK(B > 0 && N > 0 && D > 0, "Invalid dimensions");

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}