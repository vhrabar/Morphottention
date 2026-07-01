#ifndef MORPHOTTENTION_ATTENTION_CUH
#define MORPHOTTENTION_ATTENTION_CUH

#include <cuda_runtime.h>

#include <cuda_fp16.h>

#ifdef __CUDACC__
#include <cuda/sm120/matmul.cuh>
#include <cuda/sm120/project.cuh>
#include <cuda/sm120/smem.cuh>
#include <cuda/utils/declarations.cuh>
#include <cuda/utils/smem.cuh>
#include <cuda/utils/utils.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

namespace arch {
#if __CUDA_ARCH__ == 1200
namespace impl = sm120; // Consumer Blackwell
#else
namespace impl = sm120; // fallback
#endif
} // namespace arch
#endif // __CUDACC__

void attention_forward_kernel_launcher(const __half* X, const __half* W_phi, const __half* gate_q, const __half* gate_k,
                                       const __half* W_V, __half* out, float* lse, const int B, const int N,
                                       const int D, const int H, const int cube_m, const int head_dim_v,
                                       const float scale, cudaStream_t stream);

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* W_phi,
                                        const __half* gate_q, const __half* gate_k, const __half* W_V,
                                        const __half* out, const float* lse, __half* dX, __half* dW_phi,
                                        __half* d_gate_q, __half* d_gate_k, __half* dW_V, const int B, const int N,
                                        const int D, const int H, const int cube_m, const int head_dim_v,
                                        const float scale, cudaStream_t stream);

#endif // MORPHOTTENTION_ATTENTION_CUH
