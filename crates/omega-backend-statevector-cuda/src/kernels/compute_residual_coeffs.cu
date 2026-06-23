// Residual gradient observable coefficient builder.
//
// Replaces the captured `cuLaunchHostFunc` host callback with a
// pure-device kernel: for each measurement output `i`,
//
//     coeffs[i] = 2 · (float(predictions[i]) - y_label[i])
//
// Both `predictions` (one f64 per output, populated by the in-graph
// `pauli_z_expectation_to_slot` kernels) and `y_label` (one f32
// per output, populated host-side via memcpy_htod before each
// graph.launch) live on the device. Output `coeffs` feeds the
// `apply_diagonal_pauli_sum_via_pool` ν-init kernel that follows.
//
// Why this is faster than the host_func version:
// - No CUDA worker thread context switch (cuLaunchHostFunc runs
//   the callback on a CUDA-managed thread, which costs ~10 µs of
//   context-switch overhead per replay).
// - No predictions memcpy_dtoh + coeffs memcpy_htod inside the
//   graph (the host_func variant captured both as graph nodes).
// - Replaces 3 graph nodes (memcpy_dtoh + host_func + memcpy_htod)
//   with 1 (this kernel), reducing total cuGraphLaunch cost.

extern "C" {

__global__ void compute_residual_coeffs(
    const double* predictions,
    const float* y_label,
    float* coeffs,
    unsigned int num_outputs
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_outputs) return;
    coeffs[gid] = 2.0f * ((float)predictions[gid] - y_label[gid]);
}

} // extern "C"
