/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MORPHOTTENTION_REDUCTIONS_CUH
#define MORPHOTTENTION_REDUCTIONS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    const unsigned int mask = __activemask();

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);

    return val;
}

__device__ __forceinline__ float warpAllReduceSum(float val) {
    const unsigned int mask = __activemask();

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(mask, val, offset);

    return val;
}

__device__ __forceinline__ float warpMax(float val) {
    const unsigned int mask = __activemask();
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(mask, val, offset));
    return val;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float warp_sums[BLOCK_SIZE / 32];

    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0)
        warp_sums[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;

    if (warp_id == 0)
        val = warpReduceSum(val);

    return val;
}

#endif // MORPHOTTENTION_REDUCTIONS_CUH
