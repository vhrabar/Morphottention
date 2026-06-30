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

template <int ROWS, int CUBE_M, int WARPS, typename Proj = morph::Cube<morph::CubeProjection::Sigmoid>>
__device__ __forceinline__ void project_phi_bwd(const __half* __restrict__ x_tile,       // [ROWS, D]
                                                const __half* __restrict__ W_phi_h,      // [D, CUBE_M], ld = ldphi
                                                const __half* __restrict__ self_gate,    // [CUBE_M]
                                                const __half* __restrict__ partner_gate, // [CUBE_M]
                                                const float* __restrict__ d_code,        // [ROWS, CUBE_M]
                                                const float* __restrict__ d_bias,        // [ROWS]
                                                float* __restrict__ dx_out,              // [ROWS, D] out
                                                __half* __restrict__ dW_phi_g,           // [D, CUBE_M] GMEM, ld=ldphi
                                                __half* __restrict__ d_self_gate_g,      // [CUBE_M] GMEM
                                                __half* __restrict__ d_partner_gate_g,   // [CUBE_M] GMEM
                                                float* __restrict__ z_stage,             // [ROWS, CUBE_M] scratch
                                                __half* __restrict__ dz_h,               // [ROWS, CUBE_M] scratch
                                                float* __restrict__ dwphi_stage,         // [D, CUBE_M] scratch
                                                const int D, const int ldphi, const unsigned warp,
                                                const unsigned lane) {

    // Z = X @ W_phi
    matmul<ROWS, CUBE_M, WARPS, false>(z_stage, x_tile, W_phi_h, D, D, ldphi, CUBE_M, 1.0f);
    __syncthreads();

    //  dZ = (d_code + d_bias x partner_gate) x self_gate x sigtma'(Z)
    for (unsigned int r = warp; r < ROWS; r += WARPS) {
        for (unsigned int c = lane; c < CUBE_M; c += 32) {
            const float z = z_stage[r * CUBE_M + c];
            const float phi = Proj::project(z);
            const float g = __half2float(self_gate[c]);
            const float code = g * phi;
            const float dcode = d_code[r * CUBE_M + c] + d_bias[r] * __half2float(partner_gate[c]);

            // self-g @ Phi
            atomicAdd(&d_self_gate_g[c], __float2half(dcode * phi));
            // partner-g v. bias
            atomicAdd(&d_partner_gate_g[c], __float2half(d_bias[r] * code));

            dz_h[r * CUBE_M + c] = __float2half(dcode * g * Proj::grad(z));
        }
    }
    __syncthreads();

    // dW_phi += X.T @ dZ   [D, CUBE_M]
    matmul_dyn<WARPS, false, true>(dwphi_stage, x_tile, dz_h, D, CUBE_M, ROWS, D, CUBE_M, CUBE_M, 1.0f);
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < static_cast<unsigned int>(D) * CUBE_M; i += blockDim.x) {
        const unsigned int d = i / CUBE_M;
        const unsigned int c = i % CUBE_M;
        atomicAdd(&dW_phi_g[d * ldphi + c], __float2half(dwphi_stage[i]));
    }

    // dX = dZ @ W_phi.T   [ROWS, D]
    matmul_dyn<WARPS, true, false>(dx_out, dz_h, W_phi_h, ROWS, D, CUBE_M, CUBE_M, ldphi, D, 1.0f);
    __syncthreads();
}

} // namespace sm120

#endif // MORPHOTTENTION_PROJECT_CUH