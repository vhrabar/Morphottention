#include "attention.cuh"

template <int HEAD_DIM_V, int CUBE_M, int BR, int BC>
__global__ void morpho_attention_forward_kernel(const __half* __restrict__ X, const __half* __restrict__ W_phi,
                                                const __half* __restrict__ gate_q, const __half* __restrict__ gate_k,
                                                const __half* __restrict__ W_V, __half* __restrict__ out, int B, int N,
                                                int D, int H) {}

void attention_forward_kernel_launcher(const __half* X, const __half* W_phi, const __half* gate_q, const __half* gate_k,
                                       const __half* W_V, __half* out, const int B, const int N, const int D,
                                       const int H, const int cube_m, const int head_dim_v, cudaStream_t stream) {

    // checkers
    TORCH_CHECK(B > 0 && N > 0 && D > 0 && H > 0, "Invalid dimensions");
    TORCH_CHECK(D % H == 0, "D must be divisible by H");
    TORCH_CHECK(H * head_dim_v == D, "H * head_dim_v must equal D");
    TORCH_CHECK(X && W_phi && gate_q && gate_k && W_V && out, "Null pointer");

    // smem carve
    size_t smem = static_cast<size_t>(BC_BWD) * (cube_m + 1) * sizeof(__half)        //  Phi_k tile [BC, cube_m]
                  + static_cast<size_t>(BC_FWD) * (head_dim_v + 1) * sizeof(__half); //  V tile [BC, head_dim_v]

    // kernel instance
    const auto kernel = morpho_attention_forward_kernel<HEAD_DIM_V_FWD, CUBE_M_FWD, BR_FWD, BC_FWD>;

    // smem alloc check
    static int fwd_cached = -1;
    configure_kernel_smem(kernel, smem, fwd_cached, "forward_kernel");

    // launch
    dim3 grid(B * H, (N + BR_FWD - 1) / BR_FWD);
    dim3 block(BLOCK_SIZE);

    kernel<<<grid, block, smem, stream>>>(X, W_phi, gate_q, gate_k, W_V, out, B, N, D, H);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* dX, const __half* d_se,
                                        int B, int N, int D, cudaStream_t stream) {
    TORCH_CHECK(B > 0 && N > 0 && D > 0, "Invalid dimensions");

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
