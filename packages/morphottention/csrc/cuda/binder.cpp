#include "dispatch.h"
#include "registration.h"

#include <torch/library.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("forward(Tensor X, Tensor W_phi, Tensor gate_q, Tensor gate_k, Tensor W_V, "
          "int H, int cube_m, float scale, bool causal) -> Tensor[]");

    m.def("backward(Tensor grad_out, Tensor X, Tensor W_phi, Tensor gate_q, Tensor gate_k, "
          "Tensor W_V, Tensor out, Tensor lse, int H, int cube_m, float scale, bool causal) -> Tensor[]");

    m.impl("forward", torch::kCUDA, &forward);
    m.impl("backward", torch::kCUDA, &backward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)