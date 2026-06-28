#include "attention.cuh"

template <int HEAD_DIM_V, int CUBE_M, int BR, int BC>
__global__ void morpho_attention_forward_kernel(const __half* __restrict__ X, const __half* __restrict__ W_phi,
                                                const __half* __restrict__ gate_q, const __half* __restrict__ gate_k,
                                                const __half* __restrict__ W_V, __half* __restrict__ out, float* lse,
                                                int B, int N, int D, int H) {

    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    auto [t, b, tid] = get_coords();

    // shifted inputs & limits
    const unsigned int shift = b * N * D;

    const __half* X_shift = X + shift;
    const __half* W_phi_shift = W_phi + shift;
    const __half* gate_q_shift = gate_q + shift;
    const __half* gate_k_shift = gate_k + shift;
    const __half* W_V_shift = W_V + shift;
    const __half* out_shift = out + shift;
    const float* lse_shift = lse + shift;

    // smem carve
    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_mem = reinterpret_cast<__half*>(smem);        // q tile [BR, CUBE_M]
    auto* k_mem = q_mem + BR * CUBE_M;             // k tile [BC, CUBE_M] x2
    auto* v_mem = k_mem + 2 * BC * CUBE_M;         // V tile [BC, HEAD_DIM_V] x2

    auto* s_mem = reinterpret_cast<float*>(v_mem + 2 * BC * HEAD_DIM_V); // S/P fp32 tile [BR, BC]
    auto* p_h_mem = reinterpret_cast<__half*>(s_mem + BR * BC);          // P fp16 tile [BR, BC]
    auto* state_mem = reinterpret_cast<float*>(p_h_mem + BR * BC);

    float* max_mem = state_mem:
    float* lse_mem = state_mem + BR;
    float* corr_mem = state_mem + 2 * BR;
    float* cbias_mem = state_mem + 3 * BR;
    float* o_acc = cbias_mem + BC;
    float* pv_mem = s_mem;
}

void attention_forward_kernel_launcher(const __half* X, const __half* W_phi, const __half* gate_q, const __half* gate_k,
                                       const __half* W_V, __half* out, float* lse, const int B, const int N,
                                       const int D, const int H, const int cube_m, const int head_dim_v,
                                       cudaStream_t stream) {

    // checkers
    TORCH_CHECK(B > 0 && N > 0 && D > 0 && H > 0, "Invalid dimensions");
    TORCH_CHECK(D % H == 0, "D must be divisible by H");
    TORCH_CHECK(H * head_dim_v == D, "H * head_dim_v must equal D");
    TORCH_CHECK(X && W_phi && gate_q && gate_k && W_V && out, "Null pointer");

    // smem carve
    const size_t smem = sizeof(__half) * (BR_FWD * CUBE_M_FWD +         // q tile
                                          2 * BC_FWD * CUBE_M_FWD +     // k tile x2
                                          2 * BC_FWD * HEAD_DIM_V_FWD + // V tile x2
                                          BR_FWD * BC_FWD)              // P fp16 tile
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
