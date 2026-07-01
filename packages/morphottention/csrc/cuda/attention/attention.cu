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
    constexpr int XT_ROWS = (BR > BC) ? BR : BC;
    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_mem = reinterpret_cast<__half*>(smem); // q codes [BR, CUBE_M]
    auto* k_mem = q_mem + BR * CUBE_M;             // k codes [BC, CUBE_M]
    auto* v_mem = k_mem + BC * CUBE_M;             // V tile  [BC, HEAD_DIM_V]
    auto* xt_mem = v_mem + BC * HEAD_DIM_V;        // token staging [max(BR,BC), D]

    auto* s_mem = reinterpret_cast<float*>(xt_mem + XT_ROWS * D); // S/P fp32 tile [BR, BC]
    auto* p_h_mem = reinterpret_cast<__half*>(s_mem + BR * BC);   // P fp16 tile [BR, BC]
    auto* state_mem = reinterpret_cast<float*>(p_h_mem + BR * BC);

    float* max_mem = state_mem;            // [BR]
    float* lse_mem = state_mem + BR;       // [BR]
    float* corr_mem = state_mem + 2 * BR;  // [BR]
    float* qbias_mem = state_mem + 3 * BR; // [BR]
    float* cbias_mem = state_mem + 4 * BR; // [BC]
    float* o_acc = cbias_mem + BC;         // [BR, HEAD_DIM_V]
    float* pv_mem = s_mem;                 // PV scratch

    // runing vars in SMEM
    for (unsigned int i = tid; i < BR * HEAD_DIM_V; i += blockDim.x) {
        o_acc[i] = 0.0f;
    }
    for (unsigned int row = tid; row < BR; row += blockDim.x) {
        max_mem[row] = -FLT_MAX;
        lse_mem[row] = 0.0f;
    }

    // move X to SRAM
    arch::impl::smem_load(xt_mem, X_b + q_row_start * D, static_cast<unsigned int>(q_rows),
                          static_cast<unsigned int>(D), tid, BR);
    __syncthreads();
    // project Q to the gated unit-hypercube
    // q = gate_q @ sigma(W_phi.t, x_q) -> q_mem, q_bias = gate_k x q_i
    arch::impl::project_phi<BR, CUBE_M, WARPS>(xt_mem, W_phi_h, gate_q_h, gate_k_h, q_mem, qbias_mem, s_mem, D, ldphi,
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
        const int kv_block_start = static_cast<int>(tile) * BC;
        const int kv_valid = min(BC, N - kv_block_start);

        // stage K/V tokens
        arch::impl::smem_load(xt_mem, X_b + kv_block_start * D, static_cast<unsigned int>(kv_valid),
                              static_cast<unsigned int>(D), tid, BC);
        __syncthreads();

        // project K to the gated hypercube
        // k = gate_k @ sigma(W_phi.t, x_j) -> k_mem, c_bias = gate_q x k_j
        arch::impl::project_phi<BC, CUBE_M, WARPS>(xt_mem, W_phi_h, gate_k_h, gate_q_h, k_mem, cbias_mem, s_mem, D,
                                                   ldphi, warp, lane);

        // v = W_v x x_j -> s_mem -> v_curr
        arch::impl::matmul<BC, HEAD_DIM_V, WARPS, false>(s_mem, xt_mem, W_V_h, D, D, ldv, HEAD_DIM_V, 1.0f);
        __syncthreads();
        for (unsigned int i = tid; i < BC * HEAD_DIM_V; i += blockDim.x) {
            v_mem[i] = __float2half(s_mem[i]);
        }
        __syncthreads();

        // S_raw = QK.T -> s_mem
        arch::impl::matmul<BR, BC, WARPS, true>(s_mem, q_mem, k_mem, CUBE_M, CUBE_M, CUBE_M, BC, 1.0f);
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
        arch::impl::matmul<BR, HEAD_DIM_V, WARPS, false>(pv_mem, p_h_mem, v_mem, BC, BC, HEAD_DIM_V, HEAD_DIM_V, 1.0f);
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

template <int HEAD_DIM_V, int CUBE_M, int BR, int BC>
__global__ void morpho_attention_backward_kernel(const __half* __restrict__ grad_out, const __half* __restrict__ X,
                                                 const __half* __restrict__ W_phi, const __half* __restrict__ gate_q,
                                                 const __half* __restrict__ gate_k, const __half* __restrict__ W_V,
                                                 const __half* __restrict__ out, const float* __restrict__ lse,
                                                 __half* __restrict__ dX, __half* __restrict__ dW_phi,
                                                 __half* __restrict__ d_gate_q, __half* __restrict__ d_gate_k,
                                                 __half* __restrict__ dW_V, int B, int N, int D, int H, float scale) {

    static_assert(HEAD_DIM_V <= BR, "HEAD_DIM_V must be <= BR");
    static_assert(HEAD_DIM_V <= BC, "HEAD_DIM_V must be <= BC");

    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    const auto [t, b, tid] = get_coords();
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
    const __half* dO_s = grad_out + batch * N * D + head * HEAD_DIM_V;
    const __half* out_s = out + batch * N * D + head * HEAD_DIM_V;
    const float* lse_s = lse + bh * N;
    __half* dX_s = dX + batch * N * D;
    __half* dW_phi_h = dW_phi + head * CUBE_M;     // [D, CUBE_M] head slice, ld = ldphi
    __half* dW_V_h = dW_V + head * HEAD_DIM_V;     // [D, HEAD_DIM_V] head slice, ld = ldv
    __half* d_gate_q_h = d_gate_q + head * CUBE_M; // [CUBE_M]
    __half* d_gate_k_h = d_gate_k + head * CUBE_M; // [CUBE_M]

    // smem carve
    constexpr int XT_ROWS = (BR > BC) ? BR : BC;
    constexpr int DW_COLS = (CUBE_M > HEAD_DIM_V) ? CUBE_M : HEAD_DIM_V;
    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_mem = reinterpret_cast<__half*>(smem); // q codes [BR, CUBE_M]
    auto* k_mem = q_mem + BR * CUBE_M;             // k codes [BC, CUBE_M]
    auto* v_mem = k_mem + BC * CUBE_M;             // V tile  [BC, HEAD_DIM_V]
    auto* do_mem = v_mem + BC * HEAD_DIM_V;        // dO tile [BR, HEAD_DIM_V]
    auto* xt_mem = do_mem + BR * HEAD_DIM_V;       // token staging [max(BR,BC), D]
    auto* p_h_mem = xt_mem + XT_ROWS * D;          // P fp16 tile [BR, BC] (then dS_raw fp16)
    auto* dv_h = p_h_mem + BR * BC;                // dV fp16 [BC, HEAD_DIM_V]
    auto* dz_h = dv_h + BC * HEAD_DIM_V;           // dZ fp16 [max(BR,BC), CUBE_M]

    auto* s_mem = reinterpret_cast<float*>(dz_h + XT_ROWS * CUBE_M); // S/dP/dS tile [BR, BC]
    auto* dx_acc = s_mem + BR * BC;                                  // Q-row dX acc [BR, D]
    auto* dx_tmp = dx_acc + BR * D;                                  // dX_v / dX_k / dV scratch [max(BR,BC), D]
    auto* dq_acc = dx_tmp + XT_ROWS * D;                             // dq_code acc [BR, CUBE_M]
    auto* dk_code = dq_acc + BR * CUBE_M;                            // dk_code (S_raw path) [BC, CUBE_M]
    auto* z_stage = dk_code + BC * CUBE_M;                           // project_phi_bwd scratch [max(BR,BC), CUBE_M]
    auto* dwphi_stage = z_stage + XT_ROWS * CUBE_M;                  // dW_phi / dW_V tile scratch [D, DW_COLS]
    auto* state_mem = dwphi_stage + D * DW_COLS;

    float* lse_mem = state_mem;                  // [BR]
    float* delta_mem = state_mem + BR;           // [BR]
    float* qbias_mem = state_mem + 2 * BR;       // [BR]
    float* dqbias_mem = state_mem + 3 * BR;      // [BR]
    float* cbias_mem = state_mem + 4 * BR;       // [BC]
    float* dcbias_mem = state_mem + 4 * BR + BC; // [BC]

    for (unsigned int i = tid; i < BR * D; i += blockDim.x) {
        dx_acc[i] = 0.0f;
    }
    for (unsigned int i = tid; i < BR * CUBE_M; i += blockDim.x) {
        dq_acc[i] = 0.0f;
    }
    for (unsigned int i = tid; i < BR; i += blockDim.x) {
        dqbias_mem[i] = 0.0f;
    }

    // move X to SRAM
    arch::impl::smem_load(xt_mem, X_b + q_row_start * D, static_cast<unsigned int>(q_rows),
                          static_cast<unsigned int>(D), tid, BR);
    __syncthreads();

    // project Q to the gated unit-hypercube
    // q = gate_q @ sigma(W_phi.t, x_q) -> q_mem, q_bias = gate_k x q_i
    arch::impl::project_phi<BR, CUBE_M, WARPS>(xt_mem, W_phi_h, gate_q_h, gate_k_h, q_mem, qbias_mem, s_mem, D, ldphi,
                                               warp, lane);

    // stage dO -> do_mem
    for (unsigned int i = tid; i < BR * HEAD_DIM_V; i += blockDim.x) {
        const unsigned int r = i / HEAD_DIM_V;
        const unsigned int d = i % HEAD_DIM_V;
        const int g_row = q_row_start + static_cast<int>(r);
        do_mem[i] = (static_cast<int>(r) < q_rows && g_row < N) ? dO_s[g_row * D + d] : __float2half(0.0f);
    }
    __syncthreads();

    // lse_i and delta_i = sum_d (dO_i * O_i)
    for (unsigned int row = 0; row < BR / WARPS; row++) {
        const unsigned int row_cor = row + warp * (BR / WARPS);
        const int g_row = q_row_start + static_cast<int>(row_cor);
        const bool valid = (static_cast<int>(row_cor) < q_rows) && (g_row < N);

        float d_acc = 0.0f;
        if (valid) {
            for (unsigned int d = lane; d < HEAD_DIM_V; d += 32) {
                d_acc += __half2float(do_mem[row_cor * HEAD_DIM_V + d]) * __half2float(out_s[g_row * D + d]);
            }
        }
        d_acc = warpAllReduceSum(d_acc);
        if (lane == 0) {
            delta_mem[row_cor] = d_acc;
            lse_mem[row_cor] = valid ? lse_s[g_row] : -FLT_MAX;
        }
    }
    __syncthreads();

    // restore S/P from LSE
    unsigned int n_tiles = (static_cast<unsigned int>(N) + BC - 1) / BC;
    const int q_global_max = q_row_start + q_rows - 1;
    if (CAUSAL) {
        n_tiles = min(n_tiles, static_cast<unsigned int>(q_global_max) / BC + 1u);
    }

    const float scale_log2 = scale * LOG2E;

    for (unsigned int tile = 0; tile < n_tiles; tile++) {
        const int kv_block_start = static_cast<int>(tile) * BC;
        const int kv_valid = min(BC, N - kv_block_start);

        // stage K/V tokens
        arch::impl::smem_load(xt_mem, X_b + kv_block_start * D, static_cast<unsigned int>(kv_valid),
                              static_cast<unsigned int>(D), tid, BC);
        __syncthreads();

        // project K to the gated hypercube
        // k = gate_k @ sigma(W_phi.t, x_j) -> k_mem, c_bias = gate_q x k_j
        arch::impl::project_phi<BC, CUBE_M, WARPS>(xt_mem, W_phi_h, gate_k_h, gate_q_h, k_mem, cbias_mem, s_mem, D,
                                                   ldphi, warp, lane);

        // v = W_v x x_j -> s_mem -> v_mem
        arch::impl::matmul<BC, HEAD_DIM_V, WARPS, false>(s_mem, xt_mem, W_V_h, D, D, ldv, HEAD_DIM_V, 1.0f);
        __syncthreads();
        for (unsigned int i = tid; i < BC * HEAD_DIM_V; i += blockDim.x) {
            v_mem[i] = __float2half(s_mem[i]);
        }
        __syncthreads();

        // S_raw = QK.T -> s_mem
        arch::impl::matmul<BR, BC, WARPS, true>(s_mem, q_mem, k_mem, CUBE_M, CUBE_M, CUBE_M, BC, 1.0f);
        __syncthreads();

        // P = exp((2S_raw − qbias[i] − cbias[j])·scale_log2 − lse[i])
        // symetry -> -inf mask
        for (unsigned int row = 0; row < BR / WARPS; row++) {
            const unsigned int row_cor = row + warp * (BR / WARPS);
            const int q_global = q_row_start + static_cast<int>(row_cor);
            const float lse_i = lse_mem[row_cor];

            for (unsigned int c = lane; c < BC; c += 32) {
                const float s = (2.0f * s_mem[row_cor * BC + c] - qbias_mem[row_cor] - cbias_mem[c]) * scale_log2;
                const int k_global = kv_block_start + static_cast<int>(c);

                bool drop = (static_cast<int>(c) >= kv_valid) || (static_cast<int>(row_cor) >= q_rows);
                if (CAUSAL && k_global > q_global)
                    drop = true;
                if (MASK_DIAG && k_global == q_global)
                    drop = true;

                const float p = (drop || lse_i == -FLT_MAX) ? 0.0f : exp2f(s - lse_i);
                p_h_mem[row_cor * BC + c] = __float2half(p);
            }
        }
        __syncthreads();

        // dV = P.T @ dO -> dx_tmp -> dv_h
        arch::impl::matmul<BC, HEAD_DIM_V, WARPS, false, true>(dx_tmp, p_h_mem, do_mem, BR, BC, HEAD_DIM_V, HEAD_DIM_V,
                                                               1.0f);
        __syncthreads();
        for (unsigned int i = tid; i < BC * HEAD_DIM_V; i += blockDim.x) {
            dv_h[i] = __float2half(dx_tmp[i]);
        }
        __syncthreads();

        // dW_V += X.T @ dV   [D, HEAD_DIM_V]
        arch::impl::matmul_dyn<WARPS, false, true>(dwphi_stage, xt_mem, dv_h, D, HEAD_DIM_V, BC, D, HEAD_DIM_V,
                                                   HEAD_DIM_V, 1.0f);
        __syncthreads();

        for (unsigned int i = tid; i < static_cast<unsigned int>(D) * HEAD_DIM_V; i += blockDim.x) {
            const unsigned int d = i / HEAD_DIM_V;
            const unsigned int n = i % HEAD_DIM_V;
            atomicAdd(&dW_V_h[d * ldv + n], __float2half(dwphi_stage[i]));
        }

        // dX_v = dV @ W_V.T   [BC, D]
        arch::impl::matmul_dyn<WARPS, true, false>(dx_tmp, dv_h, W_V_h, BC, D, HEAD_DIM_V, HEAD_DIM_V, ldv, D, 1.0f);
        __syncthreads();

        for (unsigned int i = tid; i < BC * static_cast<unsigned int>(D); i += blockDim.x) {
            const int r = static_cast<int>(i) / D;
            const int kv_row = kv_block_start + r;
            if (r < kv_valid && kv_row < N) {
                atomicAdd(&dX_s[kv_row * D + (static_cast<int>(i) % D)], __float2half(dx_tmp[i]));
            }
        }
        __syncthreads();

        // dP = dO @ V.T -> s_mem
        arch::impl::matmul<BR, BC, WARPS, true>(s_mem, do_mem, v_mem, HEAD_DIM_V, HEAD_DIM_V, HEAD_DIM_V, BC, 1.0f);
        __syncthreads();
    }

    // store to GMEM
    for (unsigned int row = 0; row < BR / WARPS; row++) {
        const unsigned int row_cor = row + warp * (BR / WARPS);
        const int g_row = q_row_start + static_cast<int>(row_cor);
        if (static_cast<int>(row_cor) >= q_rows || g_row >= N) {
            continue;
        }
        for (unsigned int d = lane; d < static_cast<unsigned int>(D); d += 32) {
            atomicAdd(&dX_s[g_row * D + d], __float2half(dx_acc[row_cor * D + d]));
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
    const size_t smem = sizeof(__half) * (BR_FWD * CUBE_M_FWD +      // q codes
                                          BC_FWD * CUBE_M_FWD +      // k codes
                                          BC_FWD * HEAD_DIM_V_FWD +  // V tile
                                          BR_FWD * BC_FWD)           // P fp16 tile
                        + sizeof(__half) * (XT_ROWS * D)             // shared raw-token staging tile
                        + sizeof(float) * (BR_FWD * BC_FWD +         // S/raw fp32 tile
                                           4 * BR_FWD +              // max + lse + corr + qbias
                                           BC_FWD +                  // column bias
                                           BR_FWD * HEAD_DIM_V_FWD); // O accumulator

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

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* W_phi,
                                        const __half* gate_q, const __half* gate_k, const __half* W_V,
                                        const __half* out, const float* lse, __half* dX, __half* dW_phi,
                                        __half* d_gate_q, __half* d_gate_k, __half* dW_V, const int B, const int N,
                                        const int D, const int H, const int cube_m, const int head_dim_v,
                                        const float scale, cudaStream_t stream) {

    // checkers
    TORCH_CHECK(B > 0 && N > 0 && D > 0 && H > 0, "Invalid dimensions");
    TORCH_CHECK(D % H == 0, "D must be divisible by H");
    TORCH_CHECK(H * head_dim_v == D, "H * head_dim_v must equal D");
    TORCH_CHECK(cube_m == CUBE_M_FWD && head_dim_v == HEAD_DIM_V_FWD, "kernel built for fixed (cube_m, head_dim_v)");
    TORCH_CHECK(D % 16 == 0, "D must be a multiple of 16 (WMMA tile) for the backward GEMMs");
    TORCH_CHECK(grad_out && X && W_phi && gate_q && gate_k && W_V && out && lse, "Null input pointer");
    TORCH_CHECK(dX && dW_phi && d_gate_q && d_gate_k && dW_V, "Null grad pointer");

    // smem carve
    constexpr int XT_ROWS_BWD = (BR_BWD > BC_BWD) ? BR_BWD : BC_BWD;
    constexpr int DW_COLS_BWD = (CUBE_M_FWD > HEAD_DIM_V_FWD) ? CUBE_M_FWD : HEAD_DIM_V_FWD;
    const size_t smem = sizeof(__half) * (BR_BWD * CUBE_M_FWD +       // q codes
                                          BC_BWD * CUBE_M_FWD +       // k codes
                                          BC_BWD * HEAD_DIM_V_FWD +   // V tile
                                          BR_BWD * HEAD_DIM_V_FWD +   // dO tile
                                          BR_BWD * BC_BWD +           // P / dS_raw tile
                                          BC_BWD * HEAD_DIM_V_FWD +   // dV tile
                                          XT_ROWS_BWD * CUBE_M_FWD)   // dZ tile
                        + sizeof(__half) * (XT_ROWS_BWD * D)          // shared raw-token staging tile
                        + sizeof(float) * (BR_BWD * BC_BWD +          // S/dP/dS tile
                                           BR_BWD * D +               // dX_q accumulator
                                           XT_ROWS_BWD * D +          // dX_v / dX_k / dV scratch
                                           BR_BWD * CUBE_M_FWD +      // dq_code accumulator
                                           BC_BWD * CUBE_M_FWD +      // dk_code
                                           XT_ROWS_BWD * CUBE_M_FWD + // project_phi_bwd Z scratch
                                           D * DW_COLS_BWD +          // dW_phi / dW_V tile scratch
                                           4 * BR_BWD +               // lse + delta + qbias + d_qbias
                                           2 * BC_BWD);               // cbias + d_cbias

    // kernel instance
    const auto kernel = morpho_attention_backward_kernel<HEAD_DIM_V_FWD, CUBE_M_FWD, BR_BWD, BC_BWD>;

    // smem alloc check
    static int bwd_cached = -1;
    configure_kernel_smem(kernel, smem, bwd_cached, "backward_kernel");

    // launch
    dim3 grid(B * H, (N + BR_BWD - 1) / BR_BWD);
    dim3 block(BLOCK_SIZE);

    kernel<<<grid, block, smem, stream>>>(grad_out, X, W_phi, gate_q, gate_k, W_V, out, lse, dX, dW_phi, d_gate_q,
                                          d_gate_k, dW_V, B, N, D, H, scale);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}