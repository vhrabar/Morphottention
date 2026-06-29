#include "dispatch.h"

std::vector<torch::Tensor> forward(const torch::Tensor& X, const torch::Tensor& W_phi, const torch::Tensor& gate_q,
                                   const torch::Tensor& gate_k, const torch::Tensor& W_V, const int64_t H,
                                   const int64_t cube_m, const double scale, const bool causal) {
    return morpho_forward(X, W_phi, gate_q, gate_k, W_V, H, cube_m, scale, causal);
}

std::vector<torch::Tensor> backward(const torch::Tensor& grad_out, const torch::Tensor& X) {
    TORCH_CHECK(grad_out.is_cuda() && X.is_cuda(), "Gradient output and X must be a CUDA tensors");

    return morpho_backward(X, grad_out);
}