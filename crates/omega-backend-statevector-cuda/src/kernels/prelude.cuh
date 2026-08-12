// Precision prelude, prepended to every kernel by `kernels.rs`.
//
// The kernels are written once in terms of `real` / `real2` and compiled TWICE
// by NVRTC: once with the defaults below (f32) and once with
// `-DOMEGA_REAL=double -DOMEGA_REAL2=double2 -DOMEGA_MAKE_REAL2=make_double2`.
//
// This works because the kernel math uses no precision-specific intrinsics —
// no `rsqrtf`, no `__fmul_rn`, only arithmetic — so the only float-typed things
// in the sources are type names, the `make_*2` constructor, and literals.
#ifndef OMEGA_REAL
#define OMEGA_REAL float
#endif
#ifndef OMEGA_REAL2
#define OMEGA_REAL2 float2
#endif
#ifndef OMEGA_MAKE_REAL2
#define OMEGA_MAKE_REAL2 make_float2
#endif
typedef OMEGA_REAL real;
typedef OMEGA_REAL2 real2;
#define make_real2 OMEGA_MAKE_REAL2
