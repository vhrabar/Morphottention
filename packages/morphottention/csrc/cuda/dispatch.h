#ifndef MORPHOTTENTION_DISPATCH_H
#define MORPHOTTENTION_DISPATCH_H

#include <torch/extension.h>

// py-facing dispatchers
std::vector<torch::Tensor> forward(const torch::Tensor& X);

std::vector<torch::Tensor> backward(const torch::Tensor& grad_out, const torch::Tensor& X);

// CUDA-facing  dispatchers

std::vector<torch::Tensor> morpho_forward(const torch::Tensor& X);

std::vector<torch::Tensor> morpho_backward(const torch::Tensor& grad_out, const torch::Tensor& X);

#endif // MORPHOTTENTION_DISPATCH_H
