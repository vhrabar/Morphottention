#ifndef MORPHOTTENTION_SM120_MATMUL_CUH
#define MORPHOTTENTION_SM120_MATMUL_CUH

#include <cuda/utils/declarations.cuh>

#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <mma.h>

#include <type_traits>

namespace sm120 {

template <int M, int N, int WARPS, bool TRANS_B>
__device__ __forceinline__ void matmul(float* __restrict__ C, const __half* __restrict__ A,
                                       const __half* __restrict__ B, const int K, const int lda, const int ldb,
                                       const int ldc, const float alpha) {
    using b_layout = std::conditional_t<TRANS_B, nvcuda::wmma::col_major, nvcuda::wmma::row_major>;

    const unsigned int warp = threadIdx.x / 32;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, b_layout> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> acc;

    constexpr int NT = N / WN;
    constexpr int TOTAL = (M / WM) * NT;

    for (unsigned int mn = warp; mn < TOTAL; mn += WARPS) {
        const unsigned int m = mn / NT;
        const unsigned int n = mn % NT;
        nvcuda::wmma::fill_fragment(acc, 0.0f);

        for (int kt = 0; kt < K / WK; ++kt) {
            nvcuda::wmma::load_matrix_sync(frag_a, A + (m * WM) * lda + kt * WK, lda);
            if constexpr (TRANS_B) {
                // B[N, K]
                nvcuda::wmma::load_matrix_sync(frag_b, B + (n * WN) * ldb + kt * WK, ldb);
            } else {
                // B[K, N]
                nvcuda::wmma::load_matrix_sync(frag_b, B + (kt * WK) * ldb + n * WN, ldb);
            }
            nvcuda::wmma::mma_sync(acc, frag_a, frag_b, acc);
        }
        if (alpha != 1.0f) {
#pragma unroll
            for (float& v : acc.x) {
                v *= alpha;
            }
        }
        nvcuda::wmma::store_matrix_sync(C + (m * WM) * ldc + n * WN, acc, ldc, nvcuda::wmma::mem_row_major);
    }
}

} // namespace sm120

#endif // MORPHOTTENTION_SM120_MATMUL_CUH
