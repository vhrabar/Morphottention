#ifndef MORPHOTTENTION_PROJECT_CUH
#define MORPHOTTENTION_PROJECT_CUH

#include "cuda/morfology/cube.cuh"
#include "cuda/sm120/matmul.cuh"
#include "cuda/utils/declarations.cuh"
#include "cuda/utils/reductions.cuh"

#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <mma.h>

namespace sm120 {

template <int ROWS, int CUBE_M, int WARPS, typename Proj = morph::Cube<morph::CubeProjection::Sigmoid>>
__device__ __forceinline__ void project_phi(const __half* __restrict__ x_tile,       // [ROWS, D]
                                            const __half* __restrict__ W_phi_h,      // [D, CUBE_M], ld = ldphi
                                            const __half* __restrict__ self_gate,    // [CUBE_M]
                                            const __half* __restrict__ partner_gate, // [CUBE_M]
                                            __half* __restrict__ code,               // [ROWS, CUBE_M] out
                                            float* __restrict__ bias,                // [ROWS] out
                                            float* __restrict__ fp32_stage,          // [ROWS, CUBE_M] scratch
                                            const int D, const int ldphi, const unsigned warp, const unsigned lane) {

    // Z[ROWS, CUBE_M] = X @ W_phi_h
    matmul<ROWS, CUBE_M, WARPS, /*TRANS_B=*/false>(fp32_stage, x_tile, W_phi_h, D, D, ldphi, CUBE_M, 1.0f);
    __syncthreads();

    for (unsigned int r = warp; r < ROWS; r += WARPS) {
        float b_acc = 0.0f;
        for (unsigned int c = lane; c < CUBE_M; c += 32) {
            // sigma (z)
            const float phi = Proj::project(fp32_stage[r * CUBE_M + c]);
            // gate @ Phi
            const float g = __half2float(self_gate[c]) * phi;
            code[r * CUBE_M + c] = __float2half(g);
            // Sigma_c w_c x Phi_c
            b_acc += __half2float(partner_gate[c]) * g;
        }
        b_acc = warpReduceSum(b_acc);
        if (lane == 0) {
            bias[r] = b_acc;
        }
    }
    __syncthreads();
}

} // namespace sm120

#endif // MORPHOTTENTION_PROJECT_CUH