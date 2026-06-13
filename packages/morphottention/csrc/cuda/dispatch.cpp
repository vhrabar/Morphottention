#include "dispatch.h"

std::vector<torch::Tensor> attention_forward(const torch::Tensor& X) {
    TORCH_CHECK(X.is_cuda(), "Input tensor must be a CUDA tensor");

    return {X};
}

std::vector<torch::Tensor> attention_backward(const torch::Tensor& grad_out, const torch::Tensor& X) {
    TORCH_CHECK(grad_out.is_cuda() && X.is_cuda(), "Gradient output and X must be a CUDA tensors");

    return {grad_out};
}