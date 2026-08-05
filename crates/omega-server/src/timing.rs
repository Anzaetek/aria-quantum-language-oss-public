// SPDX-License-Identifier: Apache-2.0
//! Where the wall clock went.
//!
//! With work split across a laptop and a remote box, the first question about a
//! slow run is *"is this the network, the queue, or the simulation?"* — and the
//! fixes are completely different. Transfer-bound work wants batching;
//! compute-bound work wants a bigger machine. Without a split, an 80%-overhead
//! run and an 8%-overhead run look identical: both are just "slow".
//!
//! The server measures the phases only it can see (admission, execution,
//! response encoding) and publishes them two ways:
//!
//! * the standard **`Server-Timing`** header, which existing tooling and
//!   browser devtools already understand;
//! * per-row execution times in a batch body, because a header cannot carry
//!   256 rows and a search driver wants the per-trial cost as a scheduling
//!   signal.
//!
//! Durations come from [`Instant`] — a **monotonic** clock. Wall-clock time is
//! for timestamps only: an NTP step would otherwise produce negative or absurd
//! intervals, and a measurement nobody trusts is worse than none.

use std::time::Instant;

/// Accumulates named phases for one request.
///
/// Cheap enough to leave always on — a handful of `Instant::now()` calls per
/// request — so the numbers are there when a run turns out to be slow, rather
/// than needing a flag flipped and the problem reproduced.
#[derive(Debug)]
pub struct PhaseTimer {
    started: Instant,
    last: Instant,
    phases: Vec<(&'static str, f64)>,
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseTimer {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            last: now,
            phases: Vec::new(),
        }
    }

    /// Close the current phase under `name` and begin the next.
    pub fn mark(&mut self, name: &'static str) {
        let now = Instant::now();
        let ms = now.duration_since(self.last).as_secs_f64() * 1000.0;
        self.phases.push((name, ms));
        self.last = now;
    }

    /// Total elapsed since construction.
    pub fn total_ms(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    /// Recorded phases in order. Used by the tests that assert nothing goes
    /// unaccounted; the production paths consume the header/JSON forms.
    #[cfg(test)]
    pub fn phases(&self) -> &[(&'static str, f64)] {
        &self.phases
    }

    /// Format as a `Server-Timing` header value:
    /// `admit;dur=0.4, exec;dur=812.5, serialize;dur=31.2`.
    ///
    /// Three decimals: admission is measured in microseconds and rounding it to
    /// `0` would hide the very thing being demonstrated — that refusing a job
    /// is orders of magnitude cheaper than running one.
    pub fn server_timing_header(&self) -> String {
        self.phases
            .iter()
            .map(|(name, ms)| format!("{name};dur={ms:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Phases as JSON, for bodies where a header will not do.
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (name, ms) in &self.phases {
            map.insert(
                (*name).to_string(),
                serde_json::json!((ms * 1000.0).round() / 1000.0),
            );
        }
        map.insert(
            "total_ms".into(),
            serde_json::json!((self.total_ms() * 1000.0).round() / 1000.0),
        );
        serde_json::Value::Object(map)
    }
}

/// Times a single unit of work — one row of a batch.
pub struct RowTimer(Instant);

impl Default for RowTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl RowTimer {
    pub fn new() -> Self {
        Self(Instant::now())
    }
    pub fn finish_ms(self) -> f64 {
        let ms = self.0.elapsed().as_secs_f64() * 1000.0;
        (ms * 1000.0).round() / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_are_recorded_in_order_and_sum_to_the_total() {
        let mut t = PhaseTimer::new();
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.mark("admit");
        std::thread::sleep(std::time::Duration::from_millis(10));
        t.mark("exec");

        let names: Vec<&str> = t.phases().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["admit", "exec"]);

        // Nothing unaccounted: the phases must add up to the elapsed total.
        // A gap here would mean time is being spent somewhere unmeasured, which
        // is exactly the situation this module exists to rule out.
        let summed: f64 = t.phases().iter().map(|(_, ms)| ms).sum();
        let total = t.total_ms();
        assert!(
            (total - summed).abs() < total * 0.5 + 1.0,
            "phases {summed:.3} ms vs total {total:.3} ms — unaccounted time"
        );
        // And they measured something real, not zero.
        assert!(t.phases()[0].1 >= 4.0, "admit phase: {:?}", t.phases()[0]);
        assert!(t.phases()[1].1 >= 9.0, "exec phase: {:?}", t.phases()[1]);
    }

    #[test]
    fn server_timing_header_is_well_formed() {
        let mut t = PhaseTimer::new();
        t.mark("admit");
        t.mark("exec");
        let h = t.server_timing_header();
        // `name;dur=<number>` pairs, comma-separated — the standard shape.
        assert!(h.contains("admit;dur="), "{h}");
        assert!(h.contains("exec;dur="), "{h}");
        assert_eq!(h.matches(';').count(), 2, "{h}");
        for part in h.split(", ") {
            let (name, dur) = part.split_once(";dur=").expect("name;dur=value");
            assert!(!name.is_empty());
            dur.parse::<f64>().expect("duration must parse as a number");
        }
    }

    #[test]
    fn json_carries_every_phase_plus_the_total() {
        let mut t = PhaseTimer::new();
        t.mark("admit");
        t.mark("exec");
        let v = t.to_json();
        assert!(v.get("admit").is_some());
        assert!(v.get("exec").is_some());
        assert!(v.get("total_ms").and_then(|x| x.as_f64()).is_some());
    }

    #[test]
    fn empty_timer_produces_an_empty_header_not_junk() {
        let t = PhaseTimer::new();
        assert_eq!(t.server_timing_header(), "");
    }

    #[test]
    fn row_timer_measures_elapsed() {
        let r = RowTimer::new();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(r.finish_ms() >= 4.0);
    }
}
