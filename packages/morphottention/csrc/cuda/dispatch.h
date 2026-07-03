#ifndef MORPHOTTENTION_DISPATCH_H
#define MORPHOTTENTION_DISPATCH_H

#include <torch/torch.h>

// py-facing dispatchers
std::vector<torch::Tensor> forward(const torch::Tensor& X, const torch::Tensor& W_phi, const torch::Tensor& gate_q,
                                   const torch::Tensor& gate_k, const torch::Tensor& W_V, int64_t H, int64_t cube_m,
                                   double scale, bool causal);

std::vector<torch::Tensor> backward(const torch::Tensor& grad_out, const torch::Tensor& X, const torch::Tensor& W_phi,
                                    const torch::Tensor& gate_q, const torch::Tensor& gate_k, const torch::Tensor& W_V,
                                    const torch::Tensor& out, const torch::Tensor& lse, int64_t H, int64_t cube_m,
                                    double scale, bool causal);

// CUDA-facing  dispatchers

std::vector<torch::Tensor> morpho_forward(const torch::Tensor& X, const torch::Tensor& W_phi,
                                          const torch::Tensor& gate_q, const torch::Tensor& gate_k,
                                          const torch::Tensor& W_V, int64_t H, int64_t cube_m, double scale,
                                          bool causal);

std::vector<torch::Tensor> morpho_backward(const torch::Tensor& grad_out, const torch::Tensor& X,
                                           const torch::Tensor& W_phi, const torch::Tensor& gate_q,
                                           const torch::Tensor& gate_k, const torch::Tensor& W_V,
                                           const torch::Tensor& out, const torch::Tensor& lse, int64_t H,
                                           int64_t cube_m, double scale, bool causal);

#endif // MORPHOTTENTION_DISPATCH_H
