#ifndef MORPHOTTENTION_DISPATCH_H
#define MORPHOTTENTION_DISPATCH_H

#include <torch/extension.h>

// py-facing dispatchers
std::vector<torch::Tensor> attention_forward(const torch::Tensor& X);

std::vector<torch::Tensor> attention_backward(const torch::Tensor& grad_out, const torch::Tensor& X);

#endif // MORPHOTTENTION_DISPATCH_H
