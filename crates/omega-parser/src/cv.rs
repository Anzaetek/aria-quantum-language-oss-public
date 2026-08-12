// SPDX-License-Identifier: Apache-2.0
//! The **continuous-variable profile** of OPTICQASM.
//!
//! OPTICQASM names two disjoint families of operation, and they do not share an
//! execution model:
//!
//! | profile | gates | IR | executor |
//! |---|---|---|---|
//! | discrete-variable | `ps`, `bs_rx`, `hwp`, `pbs` | `omega-core` `Circuit` | `omega-backend-photonics`, Perceval |
//! | continuous-variable | `squeeze`, `displace`, `kerr` | [`CvProgram`] (here) | `omega-backend-cv`, piquasso |
//!
//! Before this module, [`crate::lower_to_ir`] answered `unknown photonic gate:
//! squeeze` — a false statement. `squeeze` is not unknown: piquasso implements
//! it as `Squeezing`, and this repository's own `omega-backend-cv` implements
//! `displace`, `kerr` and `phase_shift` directly. What is true is that the
//! *discrete-variable* IR cannot express it. A Fock-space operator is not a
//! permutation of qubit amplitudes, and no amount of `GateKind` variants makes
//! it one.
//!
//! So the CV profile gets its own import, and `aria-core`'s emitter finally has
//! a reader for what it writes.
//!
//! # Why `CvProgram` lives here and not in `omega-backend-cv`
//!
//! `omega-backend-cv` carries exactly one runtime dependency (`num-complex`) so
//! that it stays embeddable — see its crate docs. Putting a parser type in it
//! would drag the whole parser in. The edge therefore runs *outward*: this crate
//! defines the program, and a consumer (CLI, test, bridge) drives the backend
//! from it. `omega-backend-cv` does not depend on `omega-core` either, which is
//! the same reason CV gates must not become `omega-core` `GateKind`s.
//!
//! # Importing is not executing
//!
//! [`lower_opticqasm_cv`] accepts every CV op on any number of modes.
//! `omega-backend-cv` is single-mode and offers squeezing only as a state
//! *constructor* (`squeezed_vacuum`), so it cannot run all of them today. Those
//! are two different facts and they are kept apart deliberately: a file must not
//! become unreadable because the local executor is narrow, and D1 in
//! `PLAN-OPTICQASM-INTEGRITY.md` is precisely what happens when an executor
//! limitation gets written into the reader as if the gate did not exist.
//!
//! [`CvProgram::executable_on_builtin_cv`] reports the executor's limits
//! separately, naming piquasso for what it cannot do.

use crate::ast::{OpticGateApp, OpticParam, OpticQasmProgram, OpticQasmStmt};

/// A continuous-variable operation on one or two optical modes.
///
/// Angles and amplitudes are concrete: OPTICQASM 1.0's `symbol` rule exists in
/// the grammar but nothing binds symbols on this path, and silently resolving
/// an unbound symbol to a number is exactly the defect (`unwrap_or(0.0)`) that
/// `aria-core`'s emitter was rewritten to stop committing.
#[derive(Clone, Debug, PartialEq)]
pub enum CvOp {
    /// `ps(phi)` — rotation in phase space, `exp(i·phi·n̂)`.
    PhaseShift { mode: u32, phi: f64 },
    /// `squeeze(r, phi)` — `S(ζ)` with `ζ = r·e^{iφ}`, intended to match
    /// piquasso's `Squeezing(r, phi)`.
    ///
    /// **`r` is pinned by test; `phi` is NOT.** The cross-check fixture carries
    /// only `r`, the built-in backend offers squeezing as `squeezed_vacuum(r)`
    /// and refuses a non-zero `phi`, so nothing in this workspace would notice a
    /// sign or convention error in `phi`. Stated here as an untested intention
    /// rather than as a fact. Closing it needs a fixture case with `phi != 0`
    /// and an executor that can apply it.
    Squeeze { mode: u32, r: f64, phi: f64 },
    /// `displace(re, im)` — `D(α)` with `α = re + i·im`.
    ///
    /// Cartesian here and on `omega-backend-cv`; the fixture generator converts
    /// to piquasso's polar `Displacement(r, phi)` in exactly one place. Pinned
    /// by the cross-check (a `re`/`im` swap fails it at `coherent_re=0.5`).
    Displace { mode: u32, re: f64, im: f64 },
    /// `kerr(chi)` — `exp(i·chi·n̂²)`, matching piquasso's `Kerr(xi)`. Pinned by
    /// the cross-check on amplitudes; invisible in probabilities, which is why
    /// that comparison is on amplitudes.
    Kerr { mode: u32, chi: f64 },
    /// `bs_rx(theta, phi)` — two-mode.
    ///
    /// **The unitary is not pinned by anything here.** On the discrete-variable
    /// side the meaning is fixed to `omega-backend-photonics`
    /// (`[[cosθ, −e^{iφ}sinθ], [e^{−iφ}sinθ, cosθ]]`, Perceval-aligned and
    /// checked by `test_perceval_conventions.py`). On this side there is no
    /// executor — `executable_on_builtin_cv` always refuses it, the backend has
    /// no two-mode state — and the cross-check fixture is single-mode by
    /// construction, so no test can see a disagreement.
    ///
    /// piquasso's `Beamsplitter(theta, phi)` places the phase differently, so a
    /// future adapter that maps `theta` naively could read the same text
    /// differently from the DV lane with nothing failing. The variant exists so
    /// that a two-mode CV file *imports* rather than being rejected as
    /// unreadable; **do not** build an executor on it without first pinning the
    /// unitary against piquasso, as the DV side is pinned against Perceval.
    BeamSplitter { a: u32, b: u32, theta: f64, phi: f64 },
}

impl CvOp {
    /// The OPTICQASM spelling that produced this op.
    pub fn spelling(&self) -> &'static str {
        match self {
            CvOp::PhaseShift { .. } => "ps",
            CvOp::Squeeze { .. } => "squeeze",
            CvOp::Displace { .. } => "displace",
            CvOp::Kerr { .. } => "kerr",
            CvOp::BeamSplitter { .. } => "bs_rx",
        }
    }

    /// Modes this op touches, in the order written.
    pub fn modes(&self) -> Vec<u32> {
        match *self {
            CvOp::PhaseShift { mode, .. }
            | CvOp::Squeeze { mode, .. }
            | CvOp::Displace { mode, .. }
            | CvOp::Kerr { mode, .. } => vec![mode],
            CvOp::BeamSplitter { a, b, .. } => vec![a, b],
        }
    }
}

/// An imported CV circuit: a mode count and a straight-line op list.
#[derive(Clone, Debug, PartialEq)]
pub struct CvProgram {
    /// Total optical modes, summed over every `photon` declaration.
    pub modes: u32,
    pub ops: Vec<CvOp>,
}

impl CvProgram {
    /// Whether `omega-backend-cv` can run this program as written, and if not,
    /// why.
    ///
    /// Kept separate from import on purpose (see the module header). The limits
    /// are read off that crate's actual API, not guessed:
    ///
    /// * `FockState` is a **single** mode with a cutoff — there is no two-mode
    ///   state and therefore no beamsplitter;
    /// * it exposes `phase_shift`, `displace` and `kerr` as operations, but
    ///   squeezing only as the constructor `squeezed_vacuum(r, cutoff)`, so a
    ///   squeeze is executable only as the first op on its mode, and only with
    ///   `phi == 0`.
    ///
    /// piquasso has all of them, which is why it is named in every refusal.
    pub fn executable_on_builtin_cv(&self) -> Result<(), String> {
        if self.modes != 1 {
            return Err(format!(
                "`omega-backend-cv` is single-mode (`FockState` carries one mode and a \
                 cutoff) but this program declares {} modes. The program imported fine — \
                 this is an executor limit, not a language one. piquasso runs multi-mode \
                 CV programs.",
                self.modes
            ));
        }
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                CvOp::BeamSplitter { .. } => {
                    return Err(format!(
                        "op {i} (`bs_rx`) is two-mode; `omega-backend-cv` has no \
                         beamsplitter because it has no two-mode state. Use piquasso \
                         (`Beamsplitter`) for this program."
                    ))
                }
                CvOp::Squeeze { r: _, phi, .. } => {
                    if i != 0 {
                        return Err(format!(
                            "op {i} is `squeeze`, but `omega-backend-cv` offers squeezing \
                             only as the state constructor `squeezed_vacuum`, so it can \
                             only be the FIRST op. piquasso applies `Squeezing` at any \
                             point."
                        ));
                    }
                    if *phi != 0.0 {
                        return Err(format!(
                            "op {i} is `squeeze(r, {phi})`; `squeezed_vacuum` takes only \
                             the magnitude `r` and assumes phi = 0. piquasso's \
                             `Squeezing(r, phi)` takes both."
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Import the continuous-variable profile of an OPTICQASM program.
///
/// Refuses the polarization elements (`hwp`, `pbs`) rather than ignoring them:
/// polarization is a discrete-variable notion and a file mixing it with
/// squeezing is not a CV program. Refuses symbolic parameters for the reason in
/// the [`CvOp`] docs.
pub fn lower_opticqasm_cv(program: &OpticQasmProgram) -> Result<CvProgram, String> {
    let mut modes: u32 = 0;
    // reg -> (first optical mode, count)
    let mut regs: std::collections::HashMap<String, (u32, u32)> = std::collections::HashMap::new();
    let mut ops = Vec::new();

    for stmt in &program.statements {
        match stmt {
            OpticQasmStmt::PhotonDecl {
                name,
                size,
                polarized,
            } => {
                if *polarized {
                    return Err(format!(
                        "`photon {name}[{size}] pol;` declares a polarization register, \
                         which is a discrete-variable notion (H/V sub-modes). The CV \
                         profile has no polarization; use `lower_to_ir` for this program."
                    ));
                }
                regs.insert(name.clone(), (modes, *size));
                modes += size;
            }
            OpticQasmStmt::GateApp(app) => {
                ops.push(lower_cv_gate(app, &regs)?);
            }
        }
    }

    Ok(CvProgram { modes, ops })
}

fn lower_cv_gate(
    app: &OpticGateApp,
    regs: &std::collections::HashMap<String, (u32, u32)>,
) -> Result<CvOp, String> {
    let resolve = |i: usize| -> Result<u32, String> {
        let m = app
            .modes
            .get(i)
            .ok_or_else(|| format!("`{}` names no mode at position {i}", app.name))?;
        let (start, size) = regs
            .get(&m.reg)
            .ok_or_else(|| format!("undefined photon register: {}", m.reg))?;
        if m.index >= *size {
            return Err(format!(
                "mode index {} out of range for `{}[{}]`",
                m.index, m.reg, size
            ));
        }
        Ok(start + m.index)
    };

    let arity = |n: usize| -> Result<(), String> {
        if app.params.len() == n {
            Ok(())
        } else {
            Err(format!(
                "`{}` takes {n} parameter(s), got {}",
                app.name,
                app.params.len()
            ))
        }
    };

    let modes_arity = |n: usize| -> Result<(), String> {
        if app.modes.len() == n {
            Ok(())
        } else {
            Err(format!(
                "`{}` acts on {n} mode(s), got {}",
                app.name,
                app.modes.len()
            ))
        }
    };

    let p = |i: usize| -> Result<f64, String> {
        match &app.params[i] {
            OpticParam::Num(v) => Ok(*v),
            OpticParam::Symbol(s) => Err(format!(
                "`{}` has the unbound symbolic parameter `${s}`. The CV import produces \
                 concrete operators; bind it before importing rather than letting it \
                 default to a number.",
                app.name
            )),
        }
    };

    match app.name.as_str() {
        "ps" => {
            arity(1)?;
            modes_arity(1)?;
            Ok(CvOp::PhaseShift {
                mode: resolve(0)?,
                phi: p(0)?,
            })
        }
        "squeeze" => {
            arity(2)?;
            modes_arity(1)?;
            Ok(CvOp::Squeeze {
                mode: resolve(0)?,
                r: p(0)?,
                phi: p(1)?,
            })
        }
        "displace" => {
            arity(2)?;
            modes_arity(1)?;
            Ok(CvOp::Displace {
                mode: resolve(0)?,
                re: p(0)?,
                im: p(1)?,
            })
        }
        "kerr" => {
            arity(1)?;
            modes_arity(1)?;
            Ok(CvOp::Kerr {
                mode: resolve(0)?,
                chi: p(0)?,
            })
        }
        "bs_rx" | "bs" => {
            arity(2)?;
            modes_arity(2)?;
            Ok(CvOp::BeamSplitter {
                a: resolve(0)?,
                b: resolve(1)?,
                theta: p(0)?,
                phi: p(1)?,
            })
        }
        "hwp" | "pbs" => Err(format!(
            "`{}` is a polarization element and belongs to the discrete-variable \
             profile; use `lower_to_ir`.",
            app.name
        )),
        other => Err(format!(
            "unknown photonic gate: {other} (CV profile accepts ps, squeeze, displace, \
             kerr, bs_rx/bs)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_opticqasm;

    fn cv(src: &str) -> Result<CvProgram, String> {
        lower_opticqasm_cv(&parse_opticqasm(src)?)
    }

    /// The defect this module exists for: these three gates are what
    /// `aria-core::to_opticqasm` writes, and nothing in the repository could
    /// read them back.
    #[test]
    fn the_three_cv_gates_import() {
        let p = cv("OPTICQASM 1.0;\nphoton q[2];\nsqueeze(0.4, 0.2) q[0];\n\
                    displace(0.7, -0.1) q[0];\nkerr(0.15) q[1];\n")
            .expect("squeeze/displace/kerr must import");
        assert_eq!(p.modes, 2);
        assert_eq!(
            p.ops,
            vec![
                CvOp::Squeeze { mode: 0, r: 0.4, phi: 0.2 },
                CvOp::Displace { mode: 0, re: 0.7, im: -0.1 },
                CvOp::Kerr { mode: 1, chi: 0.15 },
            ],
            "parameters or modes were reordered"
        );
    }

    /// Mode resolution must be absolute across registers, not per-register —
    /// otherwise two registers both start at 0 and silently alias.
    #[test]
    fn modes_are_absolute_across_registers() {
        let p = cv("OPTICQASM 1.0;\nphoton a[2];\nphoton b[2];\nkerr(0.1) b[1];\n").unwrap();
        assert_eq!(p.modes, 4);
        assert_eq!(p.ops, vec![CvOp::Kerr { mode: 3, chi: 0.1 }]);
    }

    /// The two parameters of `squeeze` and `displace` are NOT interchangeable,
    /// so the fixture uses distinct values and the assertion above pins the
    /// order. This test pins that a swap is observable at all.
    #[test]
    fn parameter_order_is_not_symmetric() {
        let p = cv("OPTICQASM 1.0;\nphoton q[1];\ndisplace(0.7, -0.1) q[0];\n").unwrap();
        match p.ops[0] {
            CvOp::Displace { re, im, .. } => {
                assert_ne!(re, im, "fixture would be invariant under a swap");
                assert_eq!((re, im), (0.7, -0.1));
            }
            ref other => panic!("wrong op: {other:?}"),
        }
    }

    #[test]
    fn symbolic_parameter_is_refused_not_defaulted() {
        let err = cv("OPTICQASM 1.0;\nphoton q[1];\nkerr($chi) q[0];\n").unwrap_err();
        assert!(err.contains("symbolic"), "{err}");
    }

    #[test]
    fn wrong_arity_is_refused() {
        assert!(cv("OPTICQASM 1.0;\nphoton q[1];\nsqueeze(0.4) q[0];\n")
            .unwrap_err()
            .contains("2 parameter"));
        assert!(cv("OPTICQASM 1.0;\nphoton q[2];\nkerr(0.1) q[0], q[1];\n")
            .unwrap_err()
            .contains("1 mode"));
    }

    #[test]
    fn out_of_range_mode_is_refused() {
        assert!(cv("OPTICQASM 1.0;\nphoton q[2];\nkerr(0.1) q[5];\n")
            .unwrap_err()
            .contains("out of range"));
    }

    #[test]
    fn polarization_is_routed_to_the_dv_profile() {
        assert!(cv("OPTICQASM 1.0;\nphoton q[2] pol;\nkerr(0.1) q[0];\n")
            .unwrap_err()
            .contains("discrete-variable"));
        assert!(cv("OPTICQASM 1.0;\nphoton q[2];\npbs q[0], q[1];\n")
            .unwrap_err()
            .contains("discrete-variable"));
    }

    /// Import and execution are separate: all of these IMPORT, and the
    /// executor check is what reports the built-in backend's limits.
    #[test]
    fn executor_limits_are_reported_separately_from_import() {
        let multi = cv("OPTICQASM 1.0;\nphoton q[2];\nkerr(0.1) q[0];\n").unwrap();
        assert!(
            multi.executable_on_builtin_cv().unwrap_err().contains("single-mode"),
            "two modes must import but not execute on the built-in backend"
        );

        let bs = cv("OPTICQASM 1.0;\nphoton q[2];\nbs_rx(0.5, 0.1) q[0], q[1];\n").unwrap();
        assert!(bs.executable_on_builtin_cv().is_err(), "no two-mode state");

        let late = cv("OPTICQASM 1.0;\nphoton q[1];\nkerr(0.1) q[0];\nsqueeze(0.3, 0.0) q[0];\n")
            .unwrap();
        assert!(
            late.executable_on_builtin_cv().unwrap_err().contains("FIRST"),
            "squeezing is a constructor there, so it cannot follow another op"
        );

        let phi = cv("OPTICQASM 1.0;\nphoton q[1];\nsqueeze(0.3, 0.4) q[0];\n").unwrap();
        assert!(
            phi.executable_on_builtin_cv().is_err(),
            "squeezed_vacuum takes only r"
        );

        let ok = cv("OPTICQASM 1.0;\nphoton q[1];\nsqueeze(0.3, 0.0) q[0];\n\
                     displace(0.2, 0.0) q[0];\nkerr(0.1) q[0];\n")
            .unwrap();
        ok.executable_on_builtin_cv()
            .expect("single mode, squeeze first, phi=0 — this one does run");
    }
}
