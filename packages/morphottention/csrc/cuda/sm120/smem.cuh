#ifndef MORPHOTTENTION_SM120_SMEM_CUH
#define MORPHOTTENTION_SM120_SMEM_CUH

#include <cuda_runtime.h>

#include <cuda_pipeline.h>
#include <device_launch_parameters.h>
#include <mma.h>

namespace sm120 {

constexpr unsigned int CHUNK = 16;
// naive load into SMEM
__device__ __forceinline__ void smem_load(__half* __restrict__ smem, const __half* __restrict__ gmem,
                                          const unsigned int rows, const unsigned int cols, const unsigned int tid,
                                          const unsigned int padded_rows) {
    for (unsigned int i = tid; i < padded_rows * cols; i += blockDim.x) {
        smem[i] = (i < rows * cols) ? gmem[i] : __float2half(0.0f);
    }
}
// async load into SMEM
__device__ __forceinline__ void smem_load_async(__half* __restrict__ smem, const __half* __restrict__ gmem,
                                                const unsigned int rows, const unsigned int cols,
                                                const unsigned int tid, const unsigned int padded_rows) {
    const unsigned int total = padded_rows * cols;
    const unsigned int valid = rows * cols;
    for (unsigned int base = tid * CHUNK; base < total; base += blockDim.x * CHUNK) {
        const unsigned int remaining = valid - base;
        const unsigned int valid_h = (base < valid) ? (remaining < CHUNK ? remaining : CHUNK) : 0u;
        __pipeline_memcpy_async(smem + base, gmem + base, CHUNK * sizeof(__half), (CHUNK - valid_h) * sizeof(__half));
    }
}
} // namespace sm120

#endif // MORPHOTTENTION_SM120_SMEM_CUH