#ifndef MORPHOTTENTION_PROJECT_CUH
#define MORPHOTTENTION_PROJECT_CUH

#include "cuda/morfology/cube.cuh"
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
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int M_AT = ROWS / WM;
    constexpr int N_AT = CUBE_M / WN;
    const int KD_AT = D / WK;

    // Z[ROWS, CUBE_M] = X @ W_phi_h
    for (unsigned int mb = warp; mb < M_AT; mb += WARPS) {
        for (unsigned nb = 0; nb < N_AT; ++nb) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> fragment_acc;
            nvcuda::wmma::fill_fragment(fragment_acc, 0.0f);
            for (unsigned int kb = 0; kb < KD_AT; ++kb) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::row_major> fragment_a;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, nvcuda::wmma::row_major> fragment_b;

                nvcuda::wmma::load_matrix_sync(fragment_a, x_tile + (mb * WM) * D + kb * WK, D);
                nvcuda::wmma::load_matrix_sync(fragment_b, W_phi_h + (kb * WK) * ldphi + nb * WN, ldphi);
                nvcuda::wmma::mma_sync(fragment_acc, fragment_a, fragment_b, fragment_acc);
            }
            nvcuda::wmma::store_matrix_sync(fp32_stage + (mb * WM) * CUBE_M + nb * WN, fragment_acc, CUBE_M,
                                            nvcuda::wmma::mem_row_major);
        }
    }
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