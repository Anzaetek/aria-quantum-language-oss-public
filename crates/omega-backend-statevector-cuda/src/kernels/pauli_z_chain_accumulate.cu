// Stage G6 — chained Pauli-Z gradient accumulator. For a trailing
// RZ chain at the end of the forward sweep where the chain ops act
// on disjoint qubits, the adjoint identity collapses:
//
//   gradient[θ_k] = chain_k · Im⟨ν_n | Z_{q_k} | φ_n⟩
//
// for ALL k in the chain, using the SAME pre-dagger pair
// (φ_n, ν_n). Derivation: ∂RZ_k · RZ_k† = -i/2 · Z_{q_k}
// (commutator identity), and Z_{q_k} commutes with every later
// RZ_{q_j} dagger in the chain (j > k, different qubit), so the
// inner product is invariant under the daggers that would
// otherwise propagate ν_k → ν_{k-1} between gradient evaluations.
//
// This kernel computes Im⟨ν|Z_q|φ⟩ for each (q, sym, chain) entry
// in `pool[start..start+count]`, then atomic-adds chain·imag into
// `grad_dev[sym]`. Replaces the K × (DaggerPhi + DerivTriple +
// DaggerNu) = 3K graph nodes with this single launch.
//
// Grid layout: (num_chunks, count, 1). Block (chunk, k) reduces
// ν*·φ over its chunk for term k, atomic-adds chain[k] · Im(...)
// into grad_dev[sym[k]]. Each block reduces its block-local
// partial in shared memory, so multiple blocks per k feed the same
// grad_dev[sym] via atomicAdd.

extern "C" {

struct PauliZChainEntry {
    unsigned int qubit;
    unsigned int sym;
    double chain;
};

__global__ void pauli_z_chain_accumulate(
    const real2* phi,
    const real2* nu,
    double* grad_dev,
    const PauliZChainEntry* pool,
    unsigned int start,
    unsigned int count,
    unsigned long long dim
) {
    extern __shared__ real2 sdata_pz[];
    unsigned int k = blockIdx.y;
    if (k >= count) { return; }
    PauliZChainEntry entry = pool[start + k];

    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;

    real2 partial = make_real2(0.0, 0.0);
    if (gid < dim) {
        real2 phi_v = phi[gid];
        real2 nu_v = nu[gid];
        // ⟨ν|Z_q|φ⟩ contribution for this amplitude:
        //   Σ_i ν*[i] · (Z_q[i,i] · φ[i])
        // where Z_q[i,i] = +1 if bit q of i is 0, else -1.
        // ν*[i] · φ[i] = make_real2(Re, Im) where
        //   Re = ν.x·φ.x + ν.y·φ.y
        //   Im = ν.x·φ.y - ν.y·φ.x
        real sign = ((gid >> entry.qubit) & 1ULL) ? -1.0 : 1.0;
        real re = nu_v.x * phi_v.x + nu_v.y * phi_v.y;
        real im = nu_v.x * phi_v.y - nu_v.y * phi_v.x;
        partial = make_real2(sign * re, sign * im);
    }

    sdata_pz[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata_pz[threadIdx.x].x += sdata_pz[threadIdx.x + stride].x;
            sdata_pz[threadIdx.x].y += sdata_pz[threadIdx.x + stride].y;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        // Im⟨ν|Z_q|φ⟩ × chain_k → grad_dev[sym_k].
        double imag = (double)sdata_pz[0].y;
        atomicAdd(&grad_dev[entry.sym], entry.chain * imag);
    }
}

} // extern "C"
