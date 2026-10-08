// SPDX-License-Identifier: Apache-2.0
//! Read a real FCIDUMP file into a [`FermionicOp`].
//!
//! FCIDUMP is the out-of-band Hamiltonian for FermionicQASM: the circuit file
//! never names it. PySCF's `tools.fcidump` writes the bytes this parser reads
//! (`from_scf`, `from_integrals`). The header is the fixed `&FCI` namelist
//! that writer emits — `NORB`, `NELEC`, `MS2`, `ORBSYM`, `ISYM` — not a
//! general Fortran namelist, which is why this lives next to [`FermionicOp`]
//! rather than in a crate of its own.
//!
//! # Operator
//!
//! One FCIDUMP index is one fermionic mode. Index `1` in the file is mode
//! `0`. Nothing here doubles spatial orbitals into spin-orbitals: a file
//! PySCF's RHF `from_scf` wrote is the spatial skeleton it wrote, and a
//! spin-orbital Hamiltonian is a file whose indices are already
//! spin-orbitals (as `from_integrals` will write when given those tensors).
//!
//! The operator is the real chemist Hamiltonian, with the 8-fold symmetry
//! PySCF restores on read (`ao2mo.restore(..., 1)`):
//!
//! ```text
//! H = E_core
//!   + Σ_{pq} h_{pq} a†_p a_q
//!   + ½ Σ_{pqrs} (pq|rs) a†_p a†_r a_s a_q
//! ```
//!
//! `h` is symmetrised the way `fcidump.read` symmetrises it: a triangle that
//! is entirely absent is copied from the one that was stored, and a file
//! that stores both triangles is left as stored. A two-electron line and its
//! 8-fold images share one slot; the last line in the file wins, which is
//! `fcidump.read`'s rule. A line with third index `0` is one-electron when
//! the second index is nonzero and the core energy when the second index is
//! `0` — `l` is not consulted, again matching that reader.
//!
//! The read is exact. Coefficients are not dropped here; [`FermionicOp::normal_ordered`]
//! applies its own `COEFF_EPS` when a caller asks for the canonical form.
//! [`FermionicOp::jordan_wigner`] still refuses a non-Hermitian result.
//!
//! `NORB` above [`MAX_NORB`] is refused by name. Restoring every `(pq|rs)`
//! is dense in the mode count, and the Jordan–Wigner image of that sum is
//! what an expectation backend then holds.

use num_complex::Complex64;

use crate::error::{OmegaError, Result};
use crate::fermion::{FermionicOp, Ladder};

/// Largest `NORB` this reader will restore. `32^4` two-electron slots is the
/// dense sum an expectation backend can still be asked to hold; a larger
/// file is refused rather than allocated.
pub const MAX_NORB: usize = 32;

fn err(msg: impl Into<String>) -> OmegaError {
    OmegaError::Parse(format!("fcidump: {}", msg.into()))
}

fn bad_field(field: &str, raw: &str, what: &str) -> OmegaError {
    err(format!("{field}={raw}: {what}"))
}

/// Parse `text` as a real FCIDUMP file into the chemist Hamiltonian above.
pub fn parse(text: &str) -> Result<FermionicOp> {
    let lines: Vec<&str> = text.lines().collect();
    let (header, body) = split_header(&lines)?;
    let meta = parse_namelist(&header)?;
    let n = meta.norb;
    let mut h1 = vec![0.0; n * n];
    let npair = n * (n + 1) / 2;
    let mut h2 = vec![0.0; npair * (npair + 1) / 2];
    let mut ecore = 0.0;
    for line in body {
        let line = line.trim();
        if line.is_empty() {
            // `fcidump.read` stops at the first blank line.
            break;
        }
        ingest_integral(line, n, &mut h1, &mut h2, &mut ecore)?;
    }
    symmetrise_h1(&mut h1, n);
    Ok(hamiltonian(n, &h1, &h2, ecore))
}

struct Meta {
    norb: usize,
}

fn split_header<'a>(lines: &'a [&'a str]) -> Result<(Vec<String>, &'a [&'a str])> {
    if lines.is_empty() {
        return Err(err("empty file: expected an &FCI namelist"));
    }
    let mut header = Vec::new();
    for (i, line) in lines.iter().take(10).enumerate() {
        header.push((*line).to_string());
        let upper = line.to_ascii_uppercase();
        if upper.contains("&END") || upper.contains('/') {
            return Ok((header, &lines[i + 1..]));
        }
    }
    Err(err(
        "header: no &END or / in the first 10 lines (the &FCI namelist did not close)",
    ))
}

fn parse_namelist(header: &[String]) -> Result<Meta> {
    let joined = header.join(",");
    let mut tokens = joined.to_ascii_uppercase();
    if !tokens.contains("&FCI") {
        return Err(err(
            "header: missing &FCI (the namelist this reader understands)",
        ));
    }
    tokens = tokens
        .replace("&FCI", "")
        .replace("&END", "")
        .replace('/', "");
    tokens.retain(|c| !c.is_whitespace());
    // One pass, as `fcidump.read` does: `MS2=0,,ORBSYM` becomes `MS2=0,ORBSYM`.
    tokens = tokens.replace(",,", ",");

    let mut norb: Option<usize> = None;
    let mut orbsym_len: Option<usize> = None;
    for token in split_fields(&tokens) {
        if token.is_empty() {
            continue;
        }
        let Some((field, raw)) = token.split_once('=') else {
            return Err(err(format!(
                "namelist entry '{token}' has no '='; expected FIELD=value"
            )));
        };
        if field.is_empty() {
            return Err(err(format!(
                "namelist entry '{token}' has an empty field name"
            )));
        }
        match field {
            "NORB" => norb = Some(require_norb(raw)?),
            "NELEC" | "MS2" | "ISYM" => {
                require_int(field, raw)?;
            }
            "ORBSYM" => orbsym_len = Some(require_orbsym(raw)?),
            // MOLPRO and others add keys (`UHF`, …). The integrals do not
            // depend on them; a value that is present is kept only long
            // enough to know the entry was well-formed.
            _ => {}
        }
    }
    let norb = norb.ok_or_else(|| err("missing NORB"))?;
    if let Some(len) = orbsym_len {
        if len != norb {
            return Err(bad_field(
                "ORBSYM",
                &format!("{len} entries"),
                &format!("expected NORB={norb} entries"),
            ));
        }
    }
    Ok(Meta { norb })
}

/// Split on a comma that begins the next field name. `ORBSYM=1,1,1,1` stays
/// one token because those commas are followed by digits.
fn split_fields(tokens: &str) -> Vec<&str> {
    let bytes = tokens.as_bytes();
    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..bytes.len() {
        if bytes[i] == b',' {
            let next = bytes.get(i + 1).copied().unwrap_or(b' ');
            if next.is_ascii_alphabetic() {
                out.push(&tokens[start..i]);
                start = i + 1;
            }
        }
    }
    out.push(&tokens[start..]);
    out
}

fn require_int(field: &str, raw: &str) -> Result<i64> {
    let cleaned = raw.trim_end_matches(',');
    if cleaned.is_empty() {
        return Err(bad_field(field, raw, "expected an integer"));
    }
    cleaned
        .parse::<i64>()
        .map_err(|_| bad_field(field, raw, "expected an integer"))
}

fn require_norb(raw: &str) -> Result<usize> {
    let n = require_int("NORB", raw)?;
    if n < 1 {
        return Err(bad_field(
            "NORB",
            raw,
            "expected a positive integer (the number of modes)",
        ));
    }
    let n =
        usize::try_from(n).map_err(|_| bad_field("NORB", raw, "expected a positive integer"))?;
    if n > MAX_NORB {
        return Err(bad_field(
            "NORB",
            raw,
            &format!(
                "this reader restores every two-electron integral and refuses above {MAX_NORB} modes"
            ),
        ));
    }
    Ok(n)
}

fn require_orbsym(raw: &str) -> Result<usize> {
    if raw.is_empty() {
        return Err(bad_field(
            "ORBSYM",
            raw,
            "expected a comma-separated list of integers",
        ));
    }
    let mut n = 0;
    for piece in raw.split(',') {
        if piece.is_empty() {
            continue;
        }
        if piece.parse::<i64>().is_err() {
            return Err(bad_field(
                "ORBSYM",
                raw,
                "expected a comma-separated list of integers",
            ));
        }
        n += 1;
    }
    if n == 0 {
        return Err(bad_field(
            "ORBSYM",
            raw,
            "expected a comma-separated list of integers",
        ));
    }
    Ok(n)
}

fn ingest_integral(
    line: &str,
    n: usize,
    h1: &mut [f64],
    h2: &mut [f64],
    ecore: &mut f64,
) -> Result<()> {
    let cols: Vec<&str> = line.split_whitespace().collect();
    if cols.len() < 5 {
        return Err(err(format!(
            "integral line '{line}': expected a value and four indices"
        )));
    }
    let value = parse_real(cols[0]).map_err(|_| {
        err(format!(
            "integral line '{line}': value '{}' is not a real number",
            cols[0]
        ))
    })?;
    let idx = |col: usize| -> Result<i64> {
        cols[col].parse::<i64>().map_err(|_| {
            err(format!(
                "integral line '{line}': index '{}' is not an integer",
                cols[col]
            ))
        })
    };
    let (i, j, k, l) = (idx(1)?, idx(2)?, idx(3)?, idx(4)?);
    // `k == 0` is the one-electron / core distinction in `fcidump.read`.
    // `l` is ignored there, and it is ignored here.
    if k != 0 {
        let (i, j, k, l) = (
            mode_index(line, i, n)?,
            mode_index(line, j, n)?,
            mode_index(line, k, n)?,
            mode_index(line, l, n)?,
        );
        h2[chemist_index(i, j, k, l)] = value;
    } else if j != 0 {
        let (i, j) = (mode_index(line, i, n)?, mode_index(line, j, n)?);
        h1[i * n + j] = value;
    } else {
        *ecore = value;
    }
    Ok(())
}

fn parse_real(raw: &str) -> std::result::Result<f64, ()> {
    let mut s = raw.to_string();
    // Fortran dumps use `D` for the exponent. PySCF writes `e` (`%.16g`);
    // accept both so a MOLPRO file is not a parse error on the value.
    if s.contains('d') || s.contains('D') {
        s = s.replace(['d', 'D'], "e");
    }
    s.parse::<f64>().map_err(|_| ())
}

fn mode_index(line: &str, one_based: i64, n: usize) -> Result<usize> {
    if one_based < 1 || one_based > n as i64 {
        return Err(err(format!(
            "integral line '{line}': index {one_based} is outside 1..=NORB ({n})"
        )));
    }
    Ok((one_based - 1) as usize)
}

/// Pair index of chemist `(pq|`, the lower triangle including the diagonal.
/// `0`-based. Matches `i*(i-1)/2 + j - 1` on the 1-based indices PySCF uses.
fn pair_index(p: usize, q: usize) -> usize {
    let (a, b) = if p >= q { (p, q) } else { (q, p) };
    a * (a + 1) / 2 + b
}

/// Slot of `(pq|rs)` in the 8-fold packed vector `fcidump.read` returns as `H2`.
fn chemist_index(p: usize, q: usize, r: usize, s: usize) -> usize {
    let ij = pair_index(p, q);
    let kl = pair_index(r, s);
    let (a, b) = if ij >= kl { (ij, kl) } else { (kl, ij) };
    a * (a + 1) / 2 + b
}

fn symmetrise_h1(h1: &mut [f64], n: usize) {
    let lower_empty = triangle_empty(h1, n, false);
    let upper_empty = triangle_empty(h1, n, true);
    if upper_empty {
        for i in 0..n {
            for j in 0..i {
                h1[j * n + i] = h1[i * n + j];
            }
        }
    } else if lower_empty {
        for i in 0..n {
            for j in 0..i {
                h1[i * n + j] = h1[j * n + i];
            }
        }
    }
}

fn triangle_empty(h1: &[f64], n: usize, upper: bool) -> bool {
    for i in 0..n {
        for j in 0..i {
            let v = if upper { h1[j * n + i] } else { h1[i * n + j] };
            if v != 0.0 {
                return false;
            }
        }
    }
    true
}

fn hamiltonian(n: usize, h1: &[f64], h2: &[f64], ecore: f64) -> FermionicOp {
    let mut terms = Vec::new();
    if ecore != 0.0 {
        terms.push((Complex64::new(ecore, 0.0), Vec::new()));
    }
    for p in 0..n {
        for q in 0..n {
            let v = h1[p * n + q];
            if v != 0.0 {
                terms.push((
                    Complex64::new(v, 0.0),
                    vec![Ladder::raise(p as u32), Ladder::lower(q as u32)],
                ));
            }
        }
    }
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    // The ½ is the chemist convention, not a normalisation
                    // we are free to drop: (pq|rs) and (rs|pq) are both in
                    // the restored tensor, and each contributes.
                    let v = 0.5 * h2[chemist_index(p, q, r, s)];
                    if v != 0.0 {
                        terms.push((
                            Complex64::new(v, 0.0),
                            vec![
                                Ladder::raise(p as u32),
                                Ladder::raise(r as u32),
                                Ladder::lower(s as u32),
                                Ladder::lower(q as u32),
                            ],
                        ));
                    }
                }
            }
        }
    }
    FermionicOp { terms }
}
