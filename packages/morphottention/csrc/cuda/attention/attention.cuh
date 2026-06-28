#ifndef MORPHOTTENTION_ATTENTION_CUH
#define MORPHOTTENTION_ATTENTION_CUH

#include <cuda/utils/declarations.cuh>
#include <cuda/utils/smem.cuh>
#include <cuda/utils/utils.cuh>

#include <cuda_runtime.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_fp16.h>

template <int HEAD_DIM_V, int CUBE_M, int BR, int BC>
__global__ void morpho_attention_forward_kernel(const __half* __restrict__ X, const __half* __restrict__ W_phi,
                                                const __half* __restrict__ gate_q, const __half* __restrict__ gate_k,
                                                const __half* __restrict__ W_V, __half* __restrict__ out, int B, int N,
                                                int D, int H);

void attention_forward_kernel_launcher(const __half* X, const __half* W_phi, const __half* gate_q, const __half* gate_k,
                                       const __half* W_V, __half* out, const int B, const int N, const int D,
                                       const int H, const int cube_m, const int head_dim_v, cudaStream_t stream);

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* dX, const __half* d_se,
                                        int B, int N, int D, cudaStream_t stream);

#endif // MORPHOTTENTION_ATTENTION_CUH
