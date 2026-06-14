#ifndef MORPHOTTENTION_ATTENTION_CUH
#define MORPHOTTENTION_ATTENTION_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAException.h>

void attention_forward_kernel_launcher(const __half* X,  __half* out, int B, int N, int D, cudaStream_t stream);

void attention_backward_kernel_launcher(const __half* grad_out, const __half* X, const __half* dX, const __half* d_se, int B, int N, int D, cudaStream_t stream);

#endif // MORPHOTTENTION_ATTENTION_CUH
