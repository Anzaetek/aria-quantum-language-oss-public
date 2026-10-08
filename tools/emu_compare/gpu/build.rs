// SPDX-License-Identifier: Apache-2.0
fn main() {
    // The row's compiled_from stamp. Without this Cargo would not rebuild when
    // the variable changes, and a stale binary would be published against a
    // moved tree: the defect the stamp exists to catch.
    println!("cargo:rerun-if-env-changed=OMEGA_EMU_GIT_REV");
}
