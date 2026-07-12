//! Execute compiled logical programs on the omega-functions backends.
//!
//! Phase 1 exposes logical-observable read-out: given a [`PhysicalProgram`],
//! compute a logical patch's ⟨Z̄⟩ or ⟨X̄⟩ by evaluating the corresponding physical
//! Pauli string on a chosen [`SimBackend`]. Reuses the production
//! [`to_omega_core_ir`] lowering and backend selection from [`crate::ecc::run`].
//!
//! For a Clifford logical circuit the encoded state is a stabilizer state, so
//! statevector / MPS / Pauli-propagation are all exact and must agree — the
//! numeric backbone of the transversal-layer tests.

use crate::ecc::run::{to_omega_core_ir, SimBackend};
use crate::logical::compile::PhysicalProgram;
use crate::logical::patch::PauliBasis;

use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn pauli_op(b: PauliBasis) -> PauliOp {
    match b {
        PauliBasis::X => PauliOp::X,
        PauliBasis::Y => PauliOp::Y,
        PauliBasis::Z => PauliOp::Z,
    }
}

/// Expectation of a physical Pauli string on the program's circuit.
fn expect_string(
    prog: &PhysicalProgram,
    string: &[(usize, PauliBasis)],
    backend: SimBackend,
) -> Result<f64, String> {
    let obs = Observable {
        terms: vec![(
            1.0,
            string.iter().map(|&(q, b)| (q as u32, pauli_op(b))).collect(),
        )],
    };
    let ir = to_omega_core_ir(&prog.circuit);
    let be = backend.backend();
    be.expectation(&ir, &ParameterBinding::new(), &obs)
        .map_err(|e| format!("{e:?}"))
}

/// Logical ⟨Z̄⟩ of `patch`.
pub fn logical_z_expectation(
    prog: &PhysicalProgram,
    patch: usize,
    backend: SimBackend,
) -> Result<f64, String> {
    expect_string(prog, &prog.logical_z[patch], backend)
}

/// Logical ⟨X̄⟩ of `patch`.
pub fn logical_x_expectation(
    prog: &PhysicalProgram,
    patch: usize,
    backend: SimBackend,
) -> Result<f64, String> {
    expect_string(prog, &prog.logical_x[patch], backend)
}

/// Expectation of the two-patch logical-ZZ correlator ⟨Z̄_a Z̄_b⟩.
pub fn logical_zz_expectation(
    prog: &PhysicalProgram,
    a: usize,
    b: usize,
    backend: SimBackend,
) -> Result<f64, String> {
    let mut string = prog.logical_z[a].clone();
    string.extend_from_slice(&prog.logical_z[b]);
    expect_string(prog, &string, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical::compile::{compile_physical, LogicalCircuit};
    use crate::logical::transversal::SteaneTransversal;

    const TOL: f64 = 1e-9;
    const XTOL: f64 = 1e-6;

    fn prog_from(build: impl FnOnce(&mut LogicalCircuit)) -> PhysicalProgram {
        let code = SteaneTransversal::new();
        let mut lc = LogicalCircuit::new(2);
        build(&mut lc);
        compile_physical(&lc, &code)
    }

    #[test]
    fn steane_logical_zero_is_valid_codeword() {
        // prep |0>_L: ⟨Z̄⟩ = +1, ⟨X̄⟩ = 0 on the true Steane codeword.
        let prog = prog_from(|lc| {
            lc.prep_zero(0);
        });
        let zbar = logical_z_expectation(&prog, 0, SimBackend::PauliProp).unwrap();
        let xbar = logical_x_expectation(&prog, 0, SimBackend::PauliProp).unwrap();
        assert!((zbar - 1.0).abs() < TOL, "⟨Z̄⟩ = {zbar}");
        assert!(xbar.abs() < TOL, "⟨X̄⟩ = {xbar}");
    }

    #[test]
    fn steane_logical_x_flips_to_one() {
        // X̄|0>_L = |1>_L ⇒ ⟨Z̄⟩ = -1.
        let prog = prog_from(|lc| {
            lc.prep_zero(0).x(0);
        });
        let zbar = logical_z_expectation(&prog, 0, SimBackend::PauliProp).unwrap();
        assert!((zbar + 1.0).abs() < TOL, "⟨Z̄⟩ = {zbar}");
    }

    #[test]
    fn transversal_h_swaps_z_and_x() {
        // H_L|0>_L = |+>_L ⇒ ⟨X̄⟩ = +1, ⟨Z̄⟩ = 0.
        let prog = prog_from(|lc| {
            lc.prep_zero(0).h(0);
        });
        let zbar = logical_z_expectation(&prog, 0, SimBackend::PauliProp).unwrap();
        let xbar = logical_x_expectation(&prog, 0, SimBackend::PauliProp).unwrap();
        assert!(zbar.abs() < TOL, "⟨Z̄⟩ = {zbar}");
        assert!((xbar - 1.0).abs() < TOL, "⟨X̄⟩ = {xbar}");
    }

    #[test]
    fn transversal_cnot_truth_table() {
        // |c>|0> --CX--> |c>|c>. Both computational-basis cases, ⟨Z̄⟩ deterministic.
        // c = 0:
        let p00 = prog_from(|lc| {
            lc.prep_zero(0).prep_zero(1).cx(0, 1);
        });
        assert!((logical_z_expectation(&p00, 0, SimBackend::PauliProp).unwrap() - 1.0).abs() < TOL);
        assert!((logical_z_expectation(&p00, 1, SimBackend::PauliProp).unwrap() - 1.0).abs() < TOL);
        // c = 1: target flips to |1>.
        let p10 = prog_from(|lc| {
            lc.prep_zero(0).x(0).prep_zero(1).cx(0, 1);
        });
        assert!((logical_z_expectation(&p10, 0, SimBackend::PauliProp).unwrap() + 1.0).abs() < TOL);
        assert!((logical_z_expectation(&p10, 1, SimBackend::PauliProp).unwrap() + 1.0).abs() < TOL);
    }

    #[test]
    fn transversal_cnot_creates_logical_bell_correlation() {
        // |+>_c|0>_t --CX--> Bell: ⟨Z̄_c⟩ = ⟨Z̄_t⟩ = 0 but ⟨Z̄_c Z̄_t⟩ = +1.
        let prog = prog_from(|lc| {
            lc.prep_plus(0).prep_zero(1).cx(0, 1);
        });
        assert!(logical_z_expectation(&prog, 0, SimBackend::PauliProp).unwrap().abs() < TOL);
        assert!(logical_z_expectation(&prog, 1, SimBackend::PauliProp).unwrap().abs() < TOL);
        let zz = logical_zz_expectation(&prog, 0, 1, SimBackend::PauliProp).unwrap();
        assert!((zz - 1.0).abs() < TOL, "⟨Z̄_c Z̄_t⟩ = {zz}");
    }

    #[test]
    fn ideal_logical_rz_is_exact_rotation() {
        // Rz(θ)|+>_L = (|0> + e^{iθ}|1>)/√2 ⇒ ⟨X̄⟩ = cos θ, exactly (non-Clifford,
        // so read on the exact statevector backend). One patch = 7 qubits.
        use crate::logical::compile::compile_physical;
        let code = SteaneTransversal::new();
        for &theta in &[0.0_f64, std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_3] {
            let mut lc = LogicalCircuit::new(1);
            lc.prep_plus(0).rz(0, theta);
            let prog = compile_physical(&lc, &code);
            let xbar = logical_x_expectation(&prog, 0, SimBackend::Statevector).unwrap();
            assert!(
                (xbar - theta.cos()).abs() < 1e-9,
                "⟨X̄⟩ = {xbar}, expected cos({theta}) = {}",
                theta.cos()
            );
            // MPS agrees (exact for non-Clifford).
            let xmps = logical_x_expectation(&prog, 0, SimBackend::Mps).unwrap();
            assert!((xbar - xmps).abs() < 1e-6);
        }
    }

    #[test]
    fn ideal_logical_t_matches_rz_pi_over_4() {
        // Logical T ≡ Rz(π/4) up to global phase ⇒ ⟨X̄⟩ = cos(π/4).
        use crate::logical::compile::compile_physical;
        let code = SteaneTransversal::new();
        let mut lc = LogicalCircuit::new(1);
        lc.prep_plus(0).t(0);
        let prog = compile_physical(&lc, &code);
        let xbar = logical_x_expectation(&prog, 0, SimBackend::Statevector).unwrap();
        assert!(
            (xbar - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9,
            "⟨X̄⟩ = {xbar}"
        );
        assert_eq!(prog.resource.magic_states, 0); // ideal mode consumes none
    }

    #[test]
    fn faithful_t_reports_magic_and_injection() {
        // Faithful mode with a distillation protocol reports one magic state and
        // an injected logical error equal to the distilled infidelity.
        use crate::logical::compile::{compile_physical_opts, CompileOptions};
        use crate::logical::distill::{DistillProtocol, MagicStateProtocol};
        use crate::logical::transversal::NonCliffordMode;
        let code = SteaneTransversal::new();
        let magic = MagicStateProtocol::new(DistillProtocol::BravyiKitaev15to1, 1e-3, 1);
        let opts = CompileOptions {
            noncliff: NonCliffordMode::Faithful,
            magic: Some(magic),
            p_ph: 1e-3,
        };
        let mut lc = LogicalCircuit::new(1);
        lc.prep_plus(0).t(0);
        let prog = compile_physical_opts(&lc, &code, &opts);
        assert_eq!(prog.resource.magic_states, 1);
        assert!(
            (prog.resource.injected_logical_error - magic.output_infidelity()).abs() < 1e-18
        );
        // The circuit still applies the exact rotation ⇒ ⟨X̄⟩ = cos(π/4).
        let xbar = logical_x_expectation(&prog, 0, SimBackend::Statevector).unwrap();
        assert!((xbar - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn backends_agree_on_logical_clifford_circuit() {
        // A small Clifford logical circuit exercising prep/H/S/CX; statevector,
        // MPS and Pauli-propagation are all exact ⇒ ⟨Z̄⟩/⟨X̄⟩ must agree.
        let prog = prog_from(|lc| {
            lc.prep_zero(0).prep_plus(1).h(0).s(1).cx(1, 0).cz(0, 1);
        });
        for patch in 0..2 {
            let sv_z = logical_z_expectation(&prog, patch, SimBackend::Statevector).unwrap();
            let mps_z = logical_z_expectation(&prog, patch, SimBackend::Mps).unwrap();
            let pp_z = logical_z_expectation(&prog, patch, SimBackend::PauliProp).unwrap();
            assert!((sv_z - mps_z).abs() < XTOL, "sv/mps ⟨Z̄⟩ {sv_z} vs {mps_z}");
            assert!((sv_z - pp_z).abs() < XTOL, "sv/pp ⟨Z̄⟩ {sv_z} vs {pp_z}");

            let sv_x = logical_x_expectation(&prog, patch, SimBackend::Statevector).unwrap();
            let mps_x = logical_x_expectation(&prog, patch, SimBackend::Mps).unwrap();
            let pp_x = logical_x_expectation(&prog, patch, SimBackend::PauliProp).unwrap();
            assert!((sv_x - mps_x).abs() < XTOL, "sv/mps ⟨X̄⟩ {sv_x} vs {mps_x}");
            assert!((sv_x - pp_x).abs() < XTOL, "sv/pp ⟨X̄⟩ {sv_x} vs {pp_x}");
        }
    }
}
