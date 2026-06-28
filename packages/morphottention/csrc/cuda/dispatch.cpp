#include "dispatch.h"

std::vector<torch::Tensor> forward(const torch::Tensor& X) {
    TORCH_CHECK(X.is_cuda(), "Input tensor must be a CUDA tensor");

    return morpho_forward(X);
}

std::vector<torch::Tensor> backward(const torch::Tensor& grad_out, const torch::Tensor& X) {
    TORCH_CHECK(grad_out.is_cuda() && X.is_cuda(), "Gradient output and X must be a CUDA tensors");

    return morpho_backward(X, grad_out);
}