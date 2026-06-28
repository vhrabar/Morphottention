#include "dispatch.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Morphottention CUDA attention kernels";

    m.def("forward", &forward, "Attention forward dispatcher", py::arg("X"));

    m.def("backward", &backward, "Attention backward dispatcher", py::arg("grad_out"), py::arg("X"));
}
