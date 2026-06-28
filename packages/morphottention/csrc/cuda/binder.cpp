#include "dispatch.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Morphottention CUDA attention kernels";

    m.def("forward", &forward, "Attention forward dispatcher", py::arg("X"), py::arg("W_phi"), py::arg("gate_q"),
          py::arg("gate_k"), py::arg("W_V"), py::arg("H"), py::arg("cube_m"), py::arg("scale"), py::arg("causal"));

    m.def("backward", &backward, "Attention backward dispatcher", py::arg("grad_out"), py::arg("X"));
}
