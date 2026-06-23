// Shared prelude — concatenated ahead of every kernel source so the
// helpers below are defined exactly once when all kernels share an
// `ocl::Program`. Keep this file tiny: complex multiply, complex add,
// and any other primitives reused across kernels.

inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
