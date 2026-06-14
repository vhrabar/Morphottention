#include "attention.cuh"


void attention_forward_kernel_launcher(const __half* X,  __half* out, int B, int N, int D, cudaStream_t stream) {
    TORCH_CHECK(B > 0 && N > 0 && D > 0, "Invalid dimensions");

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* dX, const __half* d_se, int B, int N, int D, cudaStream_t stream) {
    TORCH_CHECK(B > 0 && N > 0 && D > 0, "Invalid dimensions");

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

