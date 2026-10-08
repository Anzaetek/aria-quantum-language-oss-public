// SPDX-License-Identifier: Apache-2.0
//! §4.4a: measuring the lane's OWN contribution, so the neighbours cancel.
//!
//! One copy, used by both `external_cpu.rs` (Linux, `/proc`) and
//! `external_cpu_ps.rs` (macOS, `ps`). It lives here rather than in each file
//! because two copies of one claim is how the absolute bar this replaces came
//! to exist in two places and get fixed in only one of them. A
//! `tests/common/mod.rs` is not built as its own test binary, so this costs no
//! extra target.

#![allow(dead_code)]

use omega_emu_compare::CpuCensus;

/// One window's worth of two censuses, so the box's other occupants cancel.
///
/// `mine` owns the lane's pids; `theirs` owns only pid 1, so everything the
/// lane does is external to *it*. Neighbours appear in **both** figures and
/// cancel in the difference, which is the lane's own contribution. That is the
/// property these tests actually want, and unlike an absolute bar on
/// `external_cores_during_row` it does not depend on the box being quiet — the
/// bar it replaces passed 5/5 and then failed at 5.50 on akilles back to back
/// with nothing changed but the neighbours.
///
/// **This relies on `tree_ticks` summing only the LISTED pids** — their own
/// `utime/stime/cutime/cstime` — and not walking live descendants. Every
/// process descends from init, so a `tree_ticks` that walked descendants would
/// make pid 1 own the whole box, `theirs` would read about zero, and the
/// difference would go **negative** rather than wrong-and-loud.
/// [`assert_attributed_to_the_lane`] pins that with a named assertion.
///
/// The two censuses are opened in one order and closed in the **same** order,
/// so the windows are the same length and merely offset, rather than one
/// containing the other. Equal lengths matter because `external_cores` is
/// Δcpu/Δwall, so unequal windows bias the difference; the returned window
/// pair is checked for it.
///
/// Returns `(contribution, ours, theirs, window_ours, window_theirs)`.
pub fn own_contribution(mine: &[u32], body: impl FnOnce()) -> (f64, f64, f64, f64, f64) {
    let theirs = CpuCensus::start(&[1]).expect("a census owning only init");
    let ours = CpuCensus::start(mine).expect("a census owning the lane");
    body();
    let t = theirs.finish().expect("theirs");
    let o = ours.finish().expect("ours");
    (
        t.external_cores_during_row - o.external_cores_during_row,
        o.external_cores_during_row,
        t.external_cores_during_row,
        o.window_s,
        t.window_s,
    )
}

/// Assert the lane's own work was attributed to the lane. Two-sided, plus the
/// two structural checks the differential depends on.
pub fn assert_attributed_to_the_lane(label: &str, n: usize, c: (f64, f64, f64, f64, f64)) {
    let (contribution, ours, theirs, w_ours, w_theirs) = c;
    let n = n as f64;

    // The windows must be comparable or the two rates are not subtractable.
    // `CpuCensus::finish` pads a short window up to `CENSUS_MIN_WINDOW_S`, so
    // a body shorter than that floor would have the FIRST finish pad and the
    // second inherit a longer window — the difference would then be partly a
    // measurement of the padding.
    assert!(
        (w_ours - w_theirs).abs() < 0.25,
        "{label}: the two censuses covered different windows ({w_ours:.2}s vs \
         {w_theirs:.2}s); their rates are not subtractable. A body shorter than \
         CENSUS_MIN_WINDOW_S does this, because the first finish pads and the second \
         inherits the padding"
    );

    // The differential inverts if pid 1 ever owns more than itself.
    assert!(
        theirs > ours,
        "{label}: owning only pid 1 gave a SMALLER external figure ({theirs:.2}) than \
         owning the lane ({ours:.2}), which inverts the differential. The likely cause \
         is `tree_ticks` having been changed to walk live descendants: every process \
         descends from init, so pid 1 would then own the whole box. This design \
         requires that it sum ONLY the listed pids"
    );

    // The floor is ONE core, not a fraction of `n`. Written as `0.75 * n`
    // first, it failed at 2.44 for n = 4 while a CI run saturated the box:
    // four threads do not get four cores on a contended machine. That floor
    // was the original defect relocated — a verdict depending on the scheduler
    // being generous rather than on the box being quiet. One core still
    // discriminates, because a census that books the lane's work as somebody
    // else's gives a contribution of about ZERO.
    assert!(
        contribution >= 1.0,
        "{label}: the lane's work contributed only {contribution:.2} cores (ours \
         {ours:.2}, theirs {theirs:.2}); a census that books the lane's own work as \
         external reads like this"
    );

    // And a ceiling, because `>= 1.0` alone is satisfied by a census that
    // attributes everything on the box to whichever pid set it is handed —
    // the `ps_total_cpu_s`-for-`ps_own_cpu_s` mutation, not a hypothetical.
    assert!(
        contribution <= 1.5 * n,
        "{label}: the lane was credited {contribution:.2} cores for at most {n} cores \
         of work (ours {ours:.2}, theirs {theirs:.2})"
    );
}
