#ifndef MORPHOTTENTION_CUBE_CUH
#define MORPHOTTENTION_CUBE_CUH

#include <cuda_runtime.h>

#include <math.h>

namespace morph {

enum class CubeProjection { Sigmoid, HardTanh01, Identity };

__device__ __forceinline__ float cube_sigmoid(const float x) {
    return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float cube_hardtanh01(const float x) {
    return fminf(1.0f, fmaxf(0.0f, 0.5f * (x + 1.0f)));
}
__device__ __forceinline__ float cube_project(const float x, CubeProjection p) {
    switch (p) {
    case CubeProjection::Sigmoid:
        return cube_sigmoid(x);
    case CubeProjection::HardTanh01:
        return cube_hardtanh01(x);
    default:
        return x;
    }
}

__device__ __forceinline__ float cube_sigmoid_grad(float x) {
    const float s = cube_sigmoid(x);
    return s * (1.0f - s);
}
__device__ __forceinline__ float cube_hardtanh01_grad(float x) {
    return (x > -1.0f && x < 1.0f) ? 0.5f : 0.0f;
}
__device__ __forceinline__ float cube_project_grad(float x, CubeProjection p) {
    switch (p) {
    case CubeProjection::Sigmoid:
        return cube_sigmoid_grad(x);
    case CubeProjection::HardTanh01:
        return cube_hardtanh01_grad(x);
    default:
        return 1.0f;
    }
}

template <CubeProjection P>
struct Cube;
template <>
struct Cube<CubeProjection::Sigmoid> {
    static __device__ __forceinline__ float project(const float x) {
        return cube_sigmoid(x);
    }
    static __device__ __forceinline__ float grad(const float x) {
        return cube_sigmoid_grad(x);
    }
};

template <>
struct Cube<CubeProjection::HardTanh01> {
    static __device__ __forceinline__ float project(const float x) {
        return cube_hardtanh01(x);
    }
    static __device__ __forceinline__ float grad(const float x) {
        return cube_hardtanh01_grad(x);
    }
};

template <>
struct Cube<CubeProjection::Identity> {
    static __device__ __forceinline__ float project(const float x) {
        return x;
    }
    static __device__ __forceinline__ float grad(float) {
        return 1.0f;
    }
};

} // namespace morph

#endif // MORPHOTTENTION_CUBE_CUH
