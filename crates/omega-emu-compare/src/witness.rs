// SPDX-License-Identifier: Apache-2.0
//! The three types that make an unwitnessed row unrepresentable *in JSON*.
//!
//! The acceptance criterion for this schema is not "the struct has a field for
//! the witness" — it is that a row missing one **cannot be written**. Putting
//! the refusal in `Serialize` rather than in a `validate()` helper is the whole
//! difference: a later lane that forgets to call the helper still gets a hard
//! error from `serde_json::to_string`, because there is no code path from an
//! absent witness to a JSON line.
//!
//! Receipt for why this is worth the three types: the A10 defect class, where
//! *the accelerated path never ran and the harness said so*
//! (`PLAN-OPEN-20260825.md:2556-2605`). A schema that accepts a row with an
//! empty witness list reproduces it with extra steps.

use serde::de::Deserializer;
use serde::ser::{Error as _, Serializer};
use serde::{Deserialize, Serialize};

/// A field whose absence voids the row (§4.1, §8).
///
/// Deliberately **not** `Option<T>`: a derived `Deserialize` fills a missing
/// `Option` field with `None` and says nothing, which is exactly the silence
/// this schema exists to remove. `Witness` has no `Default`, so a JSON line
/// without the key fails to parse, and a `Witness::absent()` in memory fails
/// to serialize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Witness<T>(Option<T>);

impl<T> Witness<T> {
    /// Record an observed witness.
    pub fn present(value: T) -> Self {
        Self(Some(value))
    }

    /// The witness was not taken. A row holding one of these does not
    /// serialize; if the measurement really cannot produce it, the honest
    /// artefact is a [`crate::VoidRow`], which says why.
    pub fn absent() -> Self {
        Self(None)
    }

    /// The witness, if it was taken.
    pub fn get(&self) -> Option<&T> {
        self.0.as_ref()
    }

    /// Whether this row can name its witness.
    pub fn is_present(&self) -> bool {
        self.0.is_some()
    }
}

impl<T> From<T> for Witness<T> {
    fn from(value: T) -> Self {
        Self::present(value)
    }
}

/// The refusal text. Serde does not tell a nested `Serialize` which field it is
/// serialising, so the message states the rule and [`crate::Row::refusals`]
/// names the fields.
pub const ABSENT_WITNESS: &str =
    "witness field absent: a row without its executed-path witness is not a row (§8); \
     publish a VoidRow saying why instead";

impl<T: Serialize> Serialize for Witness<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.0 {
            Some(value) => value.serialize(serializer),
            None => Err(S::Error::custom(ABSENT_WITNESS)),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Witness<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Witness::present)
    }
}

/// One proof that the path on the row's label is the path that ran (§4.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathWitness {
    /// What was checked, e.g. `cuda_svd_dispatch`, `quimb_max_bond`,
    /// `aer_metadata.cuStateVec_enable`, `svd_call_count`.
    pub name: String,
    /// What it read, as text, so the row carries the number a reader would
    /// want to argue with.
    pub observed: String,
    /// Whether the check reached an `assert` / a non-zero exit. §4.1 is
    /// explicit that a `println` is not a witness, so `false` refuses: the
    /// receipt is `contract_bench.rs` printing "0 of N pairs on GPU" *beside*
    /// "speedup 1.16x" and exiting 0.
    pub asserted: bool,
}

/// The per-arm witness list. Non-empty, and every entry asserted.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct PathWitnesses(Vec<PathWitness>);

impl PathWitnesses {
    /// Build a witness list. Empty lists are constructible — they just do not
    /// serialize, so the refusal lands at the one place it matters.
    pub fn new(witnesses: Vec<PathWitness>) -> Self {
        Self(witnesses)
    }

    /// The witnesses, in the order they were taken.
    pub fn as_slice(&self) -> &[PathWitness] {
        &self.0
    }

    /// Why this list is not a witness, if it is not one.
    pub fn refusal(&self) -> Option<String> {
        if self.0.is_empty() {
            return Some(
                "executed_path is empty: every arm states its precondition and \
                 refuses when it fails (§4.1); CPU-only arms witness their version \
                 pin and value gate, which is still two entries"
                    .to_string(),
            );
        }
        let printed: Vec<&str> = self
            .0
            .iter()
            .filter(|w| !w.asserted)
            .map(|w| w.name.as_str())
            .collect();
        if printed.is_empty() {
            None
        } else {
            Some(format!(
                "executed_path entries {printed:?} were printed, not asserted: \
                 §4.1 requires each witness to reach an assert or a non-zero exit"
            ))
        }
    }
}

impl Serialize for PathWitnesses {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.refusal() {
            Some(reason) => Err(S::Error::custom(reason)),
            None => self.0.serialize(serializer),
        }
    }
}

/// The revision pair of §4.4's last bullet: the rev the arm was *compiled*
/// from, embedded at build time, against the rev of the worktree it *ran* in.
///
/// > A row records the revision of the code that produced it, and refuses when
/// > it cannot. [...] a stale binary measured against a moved tree, reported as
/// > a current number.
///
/// A mismatch or a dirty tree therefore does not serialize at all. The E5
/// harness carried this as one string (`"<sha>+dirty"`) and compared it with
/// `!=`; splitting it into three typed fields closes a hole that string form
/// has, which is that an unset build stamp compares equal to itself.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitRev {
    /// `option_env!("OMEGA_EMU_GIT_REV")` at build time, in the arm binary.
    pub compiled_from: String,
    /// `git rev-parse HEAD` in the worktree, at run time.
    pub worktree: String,
    /// `git status --porcelain` was non-empty at run time.
    pub worktree_dirty: bool,
}

impl GitRev {
    /// A matched, clean pair.
    pub fn matched(rev: impl Into<String>) -> Self {
        let rev = rev.into();
        Self {
            compiled_from: rev.clone(),
            worktree: rev,
            worktree_dirty: false,
        }
    }

    /// Why this row is void, if it is.
    pub fn void_reason(&self) -> Option<String> {
        for (field, value) in [
            ("compiled_from", &self.compiled_from),
            ("worktree", &self.worktree),
        ] {
            // An unset or placeholder stamp is the hole the string form had: if
            // the build forgets to pass the rev and the runner reads the same
            // placeholder, a `!=` comparison passes while nothing is pinned.
            if value.len() != 40 || !value.chars().all(|c| c.is_ascii_hexdigit()) {
                return Some(format!(
                    "git.{field} is {value:?}, not a 40-hex commit id: the row cannot \
                     name the code that produced it (§4.4)"
                ));
            }
        }
        if self.worktree_dirty {
            return Some(
                "the worktree is dirty: the row would not name its code (§4.4)".to_string(),
            );
        }
        if self.compiled_from != self.worktree {
            return Some(format!(
                "the arm was compiled from {} but ran in a tree at {}: a stale binary \
                 measured against a moved tree is the estate's oldest recurring defect (§4.4)",
                self.compiled_from, self.worktree
            ));
        }
        None
    }
}

impl Serialize for GitRev {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(reason) = self.void_reason() {
            return Err(S::Error::custom(format!("VOID: {reason}")));
        }
        // Mirrors the derived `Deserialize` above; the round-trip test is what
        // keeps the two in step.
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("GitRev", 3)?;
        s.serialize_field("compiled_from", &self.compiled_from)?;
        s.serialize_field("worktree", &self.worktree)?;
        s.serialize_field("worktree_dirty", &self.worktree_dirty)?;
        s.end()
    }
}
