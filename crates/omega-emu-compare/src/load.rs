// SPDX-License-Identifier: Apache-2.0
//! §4.4's quiet box: the 1-minute load average above which a row is void.
//!
//! The threshold is per host on purpose. A load of 6 leaves 26 of akilles' 32
//! cores idle and saturates andromeda's 10, so one number cannot mean "quiet"
//! on both. Every lane reads the threshold from here, and the results doc
//! points here rather than carrying a copy: a predicate written in three
//! places is how the copy a test pinned stopped being the copy a publisher
//! read.

/// Threshold on a host with at most [`SMALL_HOST_MAX_CORES`] cores (andromeda).
pub const LOAD_VOID_ABOVE_SMALL_HOST: f64 = 2.0;

/// Threshold on a host with more cores than that (akilles).
pub const LOAD_VOID_ABOVE_LARGE_HOST: f64 = 8.0;

/// The largest core count that still gets the small-host threshold.
pub const SMALL_HOST_MAX_CORES: usize = 16;

/// The void threshold for a host with `cores` logical cores.
pub fn load_void_above(cores: usize) -> f64 {
    if cores <= SMALL_HOST_MAX_CORES {
        LOAD_VOID_ABOVE_SMALL_HOST
    } else {
        LOAD_VOID_ABOVE_LARGE_HOST
    }
}

/// Logical cores this process may run on. Refuses rather than guessing: a
/// wrong core count picks the wrong threshold.
///
/// This honours CPU affinity and a cgroup quota, while the load average is
/// host-wide: a lane under `taskset -c 0-7` on akilles gets 8 cores and the
/// small-host threshold against a 32-core load. That fails safe (it voids, it
/// never admits a busy box), and every lane prints the count it used so such a
/// void is explainable.
pub fn host_cores() -> Result<usize, String> {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .map_err(|e| {
            format!("cannot count this host's cores, so cannot pick §4.4's load threshold: {e}")
        })
}

/// The void threshold for the host this process runs on.
pub fn host_load_void_above() -> Result<f64, String> {
    host_cores().map(load_void_above)
}

/// A load strictly above the threshold voids the row; the threshold itself
/// does not.
pub fn void_at_load(load_1min: f64, void_above: f64) -> bool {
    load_1min > void_above
}
