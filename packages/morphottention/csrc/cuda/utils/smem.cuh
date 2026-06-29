#ifndef MORPHOTTENTION_SMEM_CUH
#define MORPHOTTENTION_SMEM_CUH

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// set SMEM size (https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity)
template <typename KernelFn>
void configure_kernel_smem(KernelFn kernel, const size_t requested, int& cached, const char* name) {
    static int device_limit = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
    if (device_limit < 0) {
        int dev = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&device_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    }
    TORCH_CHECK(static_cast<int>(requested) <= device_limit, name, " requires ", requested,
                " bytes of dynamic smem, exceeds device opt-in limit ", device_limit);
    if (static_cast<int>(requested) != cached) {
        C10_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(requested)));
        cached = static_cast<int>(requested);
    }
}

#endif // MORPHOTTENTION_SMEM_CUH
