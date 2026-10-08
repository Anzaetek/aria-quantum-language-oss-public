// SPDX-License-Identifier: Apache-2.0
fn main() {
    // The row's compiled_from stamp is this variable. Cargo does not otherwise
    // rebuild when an env var changes, which would publish a stale binary
    // against a new tree — the defect the stamp exists to catch.
    println!("cargo:rerun-if-env-changed=OMEGA_EMU_GIT_REV");
}
