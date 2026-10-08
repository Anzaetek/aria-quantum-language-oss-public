// SPDX-License-Identifier: Apache-2.0
//! Stamp the git revision into this crate at compile time.
//!
//! A comparison row has to name the revision that produced it and refuse when
//! that stamp does not match the worktree at run time. The failure that
//! prevents is a stale binary measured against a tree that has since moved,
//! published as a current number. `.git/HEAD` is watched, and so is the ref it
//! points at: on a branch HEAD contains `ref: refs/heads/...` and does not
//! itself change when a commit lands.
//!
//! A dirty tree is stamped `-dirty`. A clean stamp on a dirty tree (or the
//! reverse) is a mismatch, and the harness refuses the row.

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

fn dirty() -> bool {
    let Ok(out) = Command::new("git").args(["status", "--porcelain"]).output() else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    !out.stdout.iter().all(|b| b.is_ascii_whitespace())
}

fn main() {
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
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
    if let Some(packed) = git(&["rev-parse", "--git-path", "packed-refs"]) {
        if Path::new(&packed).exists() {
            println!("cargo:rerun-if-changed={packed}");
        }
    }

    let rev = match git(&["rev-parse", "HEAD"]) {
        Some(r) if dirty() => format!("{r}-dirty"),
        Some(r) => r,
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=OMEGA_BUILD_REV={rev}");
}
