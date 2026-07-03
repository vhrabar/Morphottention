#include "dispatch.h"

#include <torch/library.h>

TORCH_LIBRARY(morphottention, m) {
    m.def("forward(Tensor X, Tensor W_phi, Tensor gate_q, Tensor gate_k, Tensor W_V, "
          "int H, int cube_m, float scale, bool causal) -> Tensor[]");

    m.def("backward(Tensor grad_out, Tensor X, Tensor W_phi, Tensor gate_q, Tensor gate_k, "
          "Tensor W_V, Tensor out, Tensor lse, int H, int cube_m, float scale, bool causal) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(morphottention, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}