#include "attention.cuh"

#include <cuda/dispatch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

auto check = [](const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kHalf, name, " must be float16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
};

std::vector<torch::Tensor> morpho_forward(const torch::Tensor& X, const torch::Tensor& W_phi,
                                          const torch::Tensor& gate_q, const torch::Tensor& gate_k,
                                          const torch::Tensor& W_V, const int64_t H, const int64_t cube_m,
                                          const double scale, const bool causal) {
    // checkers
    check(X, "X");
    check(W_phi, "W_phi");
    check(gate_q, "gate_q");
    check(gate_k, "gate_k");
    check(W_V, "W_V");

    // shape managment
    TORCH_CHECK(X.dim() == 3, "X must be [B, N, D]");
    const int64_t B = X.size(0);
    const int64_t N = X.size(1);
    const int64_t D = X.size(2);

    TORCH_CHECK(H > 0 && D % H == 0, "D must be divisible by H");
    const int64_t head_dim_v = D / H;

    TORCH_CHECK(W_phi.dim() == 2 && W_phi.size(0) == D && W_phi.size(1) == H * cube_m, "W_phi must be [D, H*cube_m]");
    TORCH_CHECK(W_V.dim() == 2 && W_V.size(0) == D && W_V.size(1) == H * head_dim_v, "W_V must be [D, H*head_dim_v]");
    TORCH_CHECK(gate_q.dim() == 2 && gate_q.size(0) == H && gate_q.size(1) == cube_m, "gate_q must be [H, cube_m]");
    TORCH_CHECK(gate_k.dim() == 2 && gate_k.size(0) == H && gate_k.size(1) == cube_m, "gate_k must be [H, cube_m]");

    auto out = torch::empty_like(X);

    // launcher
    const c10::cuda::CUDAStreamGuard guard(c10::cuda::getCurrentCUDAStream());

    attention_forward_kernel_launcher(
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W_phi.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gate_q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gate_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W_V.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(B), static_cast<int>(N), static_cast<int>(D), static_cast<int>(H), static_cast<int>(cube_m),
        static_cast<int>(head_dim_v), c10::cuda::getCurrentCUDAStream());

    return {out};
}

std::vector<torch::Tensor> morpho_backward(const torch::Tensor& X, const torch::Tensor& grad_out) {
    TORCH_CHECK(X.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(grad_out.is_cuda(), "Gradient output tensor must be a CUDA tensor");

    auto X_contig = X.contiguous();
    auto grad_out_contig = grad_out.contiguous();

    const int B = static_cast<int>(X_contig.size(0));
    const int N = static_cast<int>(X_contig.size(1));
    const int D = static_cast<int>(X_contig.size(2));

    return {X};
}