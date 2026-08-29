// SPDX-License-Identifier: Apache-2.0
//! Stamp the git revision into the binary at compile time.
//!
//! # Why
//!
//! `omega-run` output carried no build identity, so a consumer holding two rows
//! of results could not tell whether they came from the same binary. That is
//! not hypothetical: a downstream comparison table recorded this lane twice,
//! five days apart, as the undifferentiated string "aria omega-run" — and the
//! 78x difference between the two rows turned out to be two different builds
//! rather than a regression. Nobody could see that from the table.
//!
//! # The two ways a build stamp lies, both handled here
//!
//! **A stale stamp.** Cargo caches build-script output, so without an explicit
//! `rerun-if-changed` the revision is baked at the first compile and never
//! updates. The binary then reports a commit it is not built from, which is
//! strictly worse than reporting nothing — an absent stamp prompts a question,
//! a wrong one ends it. `.git/HEAD` is watched, and so is the file HEAD
//! resolves to, because on a branch `.git/HEAD` holds `ref: refs/heads/main`
//! and does not itself change when a commit lands.
//!
//! **A dirty tree.** A stamp of `eedc80d` on a tree with uncommitted changes
//! describes a commit whose contents are not what ran. Marked `-dirty`, which
//! is the honest answer and the one a reader can act on.
//!
//! Falls back to `unknown` when git is unavailable or this is not a checkout —
//! a source tarball is a legitimate way to build, and guessing would reintroduce
//! the lying-stamp problem from the other direction.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn main() {
    // Watch the refs that decide the answer, so the stamp cannot go stale in
    // the build cache. `--git-path` resolves worktrees and submodules, where
    // `.git` is a file rather than a directory.
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
        // `.git/HEAD` on a branch is `ref: refs/heads/<name>`; the commit id
        // lives in that ref, and only it changes when a commit lands.
        if let Ok(contents) = std::fs::read_to_string(&head) {
            if let Some(r) = contents.strip_prefix("ref:") {
                if let Some(p) = git(&["rev-parse", "--git-path", r.trim()]) {
                    if Path::new(&p).exists() {
                        println!("cargo:rerun-if-changed={p}");
                    }
                }
            }
        }
    }
    // Packed refs: a freshly cloned or gc'd repo keeps branch tips here rather
    // than as loose files, so the ref path above may not exist.
    if let Some(packed) = git(&["rev-parse", "--git-path", "packed-refs"]) {
        if Path::new(&packed).exists() {
            println!("cargo:rerun-if-changed={packed}");
        }
    }

    let rev = match git(&["rev-parse", "--short", "HEAD"]) {
        Some(r) => {
            // `--porcelain` is empty exactly when the tree matches HEAD.
            let dirty = git(&["status", "--porcelain"]).is_some();
            if dirty {
                format!("{r}-dirty")
            } else {
                r
            }
        }
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=OMEGA_BUILD_REV={rev}");
}
