#pragma once

#include "cube.cuh"
#include <cuda_fp16.h>

namespace morph {

__device__ __forceinline__ float trop_add(const float s, const float v) { return s + v; }

struct SoftLSE {
    float m;
    float l;
};

__device__ __forceinline__ SoftLSE lse_init() {
    SoftLSE acc; acc.m = -INFINITY; acc.l = 0.0f; return acc;
}

__device__ __forceinline__ void lse_update(SoftLSE& acc, const float a) {
    const float m_new = fmaxf(acc.m, a);
    acc.l = acc.l * __expf(acc.m - m_new) + __expf(a - m_new);
    acc.m = m_new;
}

// fold raw (S, V) pair.
__device__ __forceinline__ void morph_accumulate(SoftLSE& acc, const float s, const float v, const float sgn, const float inv_tau) {
    lse_update(acc, sgn * trop_add(s, v) * inv_tau);
}

// Merge two SoftLSE accumulators.
__device__ __forceinline__ SoftLSE lse_merge(const SoftLSE x, const SoftLSE y) {
    if (x.m == -INFINITY) return y;
    if (y.m == -INFINITY) return x;
    const float m_new = fmaxf(x.m, y.m);
    SoftLSE r;
    r.m = m_new;
    r.l = x.l * __expf(x.m - m_new) + y.l * __expf(y.m - m_new);
    return r;
}

__device__ __forceinline__ SoftLSE lse_warp_reduce(SoftLSE acc) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        SoftLSE other;
        other.m = __shfl_down_sync(0xffffffffu, acc.m, off);
        other.l = __shfl_down_sync(0xffffffffu, acc.l, off);
        acc = lse_merge(acc, other);
    }
    return acc;
}

// out = sgn * tau * (m + log l).
__device__ __forceinline__ float lse_finalize(const SoftLSE acc, const float sgn, float tau) {
    return sgn * tau * (acc.m + __logf(acc.l));
}


//   P_ijc = exp( sgn * ((S_ij + V_jc) - out_ic) * inv_tau )
__device__ __forceinline__ float morph_weight(const float s, const float v, const float out_ic, const float sgn, const float inv_tau) {
    return __expf(sgn * (trop_add(s, v) - out_ic) * inv_tau);
}

// dtau_ic = (out_ic - sum_j P_ijc*(S_ij+V_jc)) * inv_tau
// dtau    = sum_ic dO_ic * dtau_ic
__device__ __forceinline__ float morph_dtau_term(const float P, const float s, const float v) {
    return P * trop_add(s, v);
}

} // namespace morph
