#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

std::vector<torch::Tensor> morpho_forward(const torch::Tensor& X) {
    TORCH_CHECK(X.is_cuda(), "Input tensor must be a CUDA tensor");

    auto X_contig = X.contiguous();

    const int B = static_cast<int>(X_contig.size(0));
    const int N = static_cast<int>(X_contig.size(1));
    const int D = static_cast<int>(X_contig.size(2));

    return {X};
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