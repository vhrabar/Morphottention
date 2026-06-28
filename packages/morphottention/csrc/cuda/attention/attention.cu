#include "attention.cuh"

#include <cuda_pipeline.h>

#include <cfloat>

template <int HEAD_DIM_V, int CUBE_M, int BR, int BC>
__global__ void morpho_attention_forward_kernel(const __half* __restrict__ X, const __half* __restrict__ W_phi,
                                                const __half* __restrict__ gate_q, const __half* __restrict__ gate_k,
                                                const __half* __restrict__ W_V, __half* __restrict__ out, float* lse,
                                                int B, int N, int D, int H) {

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

    const __half* X_b = X + batch * N * D;
    const __half* W_phi_h = W_phi + head * CUBE_M;
    const __half* W_V_h = W_V + head * HEAD_DIM_V;
    const __half* gate_q_h = gate_q + head * CUBE_M;
    const __half* gate_k_h = gate_k + head * CUBE_M;
    __half* out_s = out + batch * N * D + head * HEAD_DIM_V;
    float* lse_s = lse + bh * N;

    // smem carve
    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_mem = reinterpret_cast<__half*>(smem); // q tile [BR, CUBE_M]
    auto* k_mem = q_mem + BR * CUBE_M;             // k tile  [BC, CUBE_M] x2
    auto* v_mem = k_mem + 2 * BC * CUBE_M;         // V tile  [BC, HEAD_DIM_V] x2

    auto* s_mem = reinterpret_cast<float*>(v_mem + 2 * BC * HEAD_DIM_V); // S/P fp32 tile [BR, BC]
    auto* p_h_mem = reinterpret_cast<__half*>(s_mem + BR * BC);          // P fp16 tile [BR, BC]
    auto* state_mem = reinterpret_cast<float*>(p_h_mem + BR * BC);

    float* max_mem = state_mem;            // [BR]
    float* lse_mem = state_mem + BR;       // [BR]
    float* corr_mem = state_mem + 2 * BR;  // [BR]
    float* cbias_mem = state_mem + 3 * BR; // [BC]
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
    smem_load(x_q_mem, X_b + q_row_start * D, static_cast<unsigned int>(q_rows), static_cast<unsigned int>(D), tid, BR);

    const unsigned int kv0 = static_cast<unsigned int>(min(BC, N));
    smem_load_async(x_k_mem, X_b, kv0, static_cast<unsigned int>(D), tid, BC);
    __pipeline_commit();
    __syncthreads();

    // store to GMEM
    for (unsigned int row = 0; row < BR / WARPS; row++) {
        const unsigned int row_cor = row + warp * (BR / WARPS);
        const unsigned int g_row = q_row_start + row_cor;
        if (row_cor >= static_cast<unsigned int>(q_rows) || g_row >= static_cast<unsigned int>(N)) {
            continue;
        }
        for (unsigned int d = lane; d < HEAD_DIM_V; d += 32) {
            out_s[g_row * D + d] = __float2half(o_acc[row_cor * HEAD_DIM_V + d] / lse_mem[row_cor]);
        }
        if (lane == 0) {
            lse_s[g_row] = max_mem[row_cor] + log2f(lse_mem[row_cor]);
        }
    }
}

void attention_forward_kernel_launcher(const __half* X, const __half* W_phi, const __half* gate_q, const __half* gate_k,
                                       const __half* W_V, __half* out, float* lse, const int B, const int N,
                                       const int D, const int H, const int cube_m, const int head_dim_v,
                                       cudaStream_t stream) {

    // checkers
    TORCH_CHECK(B > 0 && N > 0 && D > 0 && H > 0, "Invalid dimensions");
    TORCH_CHECK(D % H == 0, "D must be divisible by H");
    TORCH_CHECK(H * head_dim_v == D, "H * head_dim_v must equal D");
    TORCH_CHECK(X && W_phi && gate_q && gate_k && W_V && out && lse, "Null pointer");

    // smem carve
    const size_t smem = sizeof(__half) * (BR_FWD * CUBE_M_FWD +         // q tile
                                          2 * BC_FWD * CUBE_M_FWD +     // k tile x2
                                          2 * BC_FWD * HEAD_DIM_V_FWD + // V tile x2
                                          BR_FWD * BC_FWD)              // P fp16 tile
                        + sizeof(__half) * (BR_FWD * D +                // X_q raw token tile
                                            2 * BC_FWD * D)             // X_k raw token tiles x2
                        + sizeof(float) * (BR_FWD * BC_FWD +            // S/P fp32 tile
                                           3 * BR_FWD +                 // max + lse + corr
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

    kernel<<<grid, block, smem, stream>>>(X, W_phi, gate_q, gate_k, W_V, out, lse, B, N, D, H);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* dX, const __half* d_se,
                                        int B, int N, int D, cudaStream_t stream) {
    TORCH_CHECK(B > 0 && N > 0 && D > 0, "Invalid dimensions");

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
