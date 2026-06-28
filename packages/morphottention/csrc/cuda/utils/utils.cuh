#ifndef MORPHOTTENTION_UTILS_CUH
#define MORPHOTTENTION_UTILS_CUH

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

// Fp32 -> fp16 casters
inline const __half* h_ptr(const at::Tensor& t) {
    return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());
}
inline __half* h_ptr(at::Tensor& t) {
    return reinterpret_cast<__half*>(t.data_ptr<at::Half>());
}

struct ThreadCoords {
    const unsigned int b;
    const unsigned int d;
    const unsigned int tid;
};

// helper boilerplate func to return CUDA dev vars
__device__ __forceinline__ ThreadCoords get_coords() {
    return {blockIdx.x, blockIdx.y, threadIdx.x};
}


#endif // MORPHOTTENTION_UTILS_CUH
