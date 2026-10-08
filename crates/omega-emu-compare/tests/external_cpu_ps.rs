// SPDX-License-Identifier: Apache-2.0
//! §4.4a's census on the macOS path, proven to fire — the same obligation
//! `external_cpu.rs` discharges for Linux.
//!
//! The Linux tests are `#![cfg(target_os = "linux")]`, so until this file
//! existed the macOS path had no test that it voids under contention. That is
//! the condition §4.4a attaches to any replacement for the load-average rule:
//! a criterion that stops voiding is not a fixed criterion but a removed one,
//! and the corrected rule would then admit exactly the disturbed rows the
//! original was written to catch. On this platform the stakes are higher, not
//! lower, because `load1` here is not a measure of CPU contention at all
//! (3.28 then 1.92 a minute apart on a box that was 85% idle), so the census
//! is the *only* criterion with any content.
//!
//! Split from `external_cpu.rs` rather than added to it so neither file needs
//! a `cfg` on every test, and so the two platforms' burners cannot be read as
//! one test file's flakiness.
#![cfg(target_os = "macos")]

use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use omega_emu_compare::{
    external_cpu_sampled, parse_ps_table, parse_ps_time, ps_own_cpu_s, ps_total_cpu_s,
    void_at_load, CpuCensus, LOAD_VOID_ABOVE_SMALL_HOST,
};

/// The two burning tests would read each other's work as external, so they
/// never overlap, whatever the harness's thread count. (Two load tests ran in
/// parallel in an earlier draft and the "quiet" one was measuring the other
/// one's spinners.)
static BOX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
mod common;
use common::{assert_attributed_to_the_lane, own_contribution};

// ---------------------------------------------------------------------------
// parsing
// ---------------------------------------------------------------------------

#[test]
fn ps_time_is_read_from_the_right() {
    assert_eq!(parse_ps_time("0:00.01"), Some(0.01));
    assert_eq!(parse_ps_time("1:30.50"), Some(90.5));
    assert_eq!(parse_ps_time("00:00:12"), Some(12.0));
    assert_eq!(parse_ps_time("2-03:04:05"), Some(183_845.0));
    assert_eq!(parse_ps_time(" 0:02.00 "), Some(2.0));
}

#[test]
fn a_long_running_process_is_not_read_as_days() {
    // launchd on andromeda prints `284:25.07`: 284 *minutes*, because macOS
    // grows the minutes field rather than adding an hours field. Read as hh:mm
    // that is 11.9 days instead of 4.7 hours, and launchd is in every census.
    let v = parse_ps_time("284:25.07").expect("mm:ss.cc");
    assert!(
        (v - (284.0 * 60.0 + 25.07)).abs() < 1e-9,
        "284:25.07 must be 284 minutes, got {v} s"
    );
    assert!(v < 86_400.0, "{v} s is over a day; the fields were misread");
}

#[test]
fn unparseable_times_are_rejected_rather_than_guessed() {
    for s in ["", "   ", "abc", "1:2:3:4:5", "-1:00", "1:xx"] {
        assert_eq!(parse_ps_time(s), None, "{s:?} parsed");
    }
}

#[test]
fn a_truncated_ps_row_is_skipped_not_fatal() {
    let rows = parse_ps_table(
        "  1 284:25.07\n\
         bad line\n\
         327 71:27.37\n\
         999\n\
         501 0:00.50\n",
    );
    assert_eq!(rows.len(), 3, "got {rows:?}");
    assert_eq!(rows[2].pid, 501);
    assert!((ps_total_cpu_s(&rows) - (17_065.07 + 4287.37 + 0.5)).abs() < 1e-6);
}

#[test]
fn own_counts_the_declared_pids_and_nothing_else() {
    // Deliberately the same semantics as the Linux path's
    // `/proc/<pid>/stat` read: every thread of a listed process, and no
    // child the lane did not declare. A row's number has to mean the same
    // thing on both boxes.
    let rows = parse_ps_table("1 0:10.00\n100 0:04.00\n101 0:08.00\n");
    assert_eq!(ps_own_cpu_s(&rows, &[100]), 4.0);
    assert_eq!(ps_own_cpu_s(&rows, &[100, 101]), 12.0);
    assert_eq!(ps_own_cpu_s(&rows, &[4242]), 0.0);
    assert_eq!(ps_total_cpu_s(&rows), 22.0);
}

// ---------------------------------------------------------------------------
// arithmetic
// ---------------------------------------------------------------------------

#[test]
fn the_arithmetic_subtracts_the_own_tree() {
    // 30 s of system CPU, 26 s of it ours, over 2 s of wall.
    let e = external_cpu_sampled(30.0, 26.0, 2.0, &[1]);
    assert!((e.external_cores_during_row - 2.0).abs() < 1e-12, "{e:?}");
    assert!(e.method.contains("ps"), "the row must name the path: {e:?}");
}

#[test]
fn the_lane_at_full_tilt_alone_on_the_box_is_not_contention() {
    // The defect §4.4a corrects, stated as arithmetic: 23 cores of our own
    // work and nothing else must read as zero external.
    let e = external_cpu_sampled(230.0, 230.0, 10.0, &[1]);
    assert_eq!(e.external_cores_during_row, 0.0);
    assert!(!void_at_load(
        e.external_cores_during_row,
        LOAD_VOID_ABOVE_SMALL_HOST
    ));
}

#[test]
fn losing_one_of_our_own_children_does_not_read_as_negative_external_load() {
    // `ps` cannot see a process that exited inside the window, so `own` can
    // overshoot the system total. Reporting that as negative external CPU
    // would read as positive evidence the box was quiet.
    let e = external_cpu_sampled(10.0, 15.0, 10.0, &[1]);
    assert_eq!(e.external_cores_during_row, 0.0);
}

// ---------------------------------------------------------------------------
// §4.4a's condition: it must still void under real external contention
// ---------------------------------------------------------------------------

/// Busy processes outside the lane's declared tree, killed on drop.
struct Burners(Vec<Child>);

impl Burners {
    fn spawn(n: usize) -> Self {
        Burners(
            (0..n)
                .map(|_| {
                    Command::new("sh")
                        .arg("-c")
                        .arg("while :; do :; done")
                        .spawn()
                        .expect("a burner")
                })
                .collect(),
        )
    }
}

impl Drop for Burners {
    fn drop(&mut self) {
        for b in &mut self.0 {
            let _ = b.kill();
            let _ = b.wait();
        }
    }
}

/// Three busy processes that are NOT in the lane's declared tree: the census
/// must see about three cores and void the row at andromeda's threshold of 2.
///
/// Three rather than the Linux file's twelve because andromeda has ten cores,
/// and the burners must stay well inside half the box or they are measuring
/// the scheduler rather than the census.
#[test]
fn someone_elses_work_voids_the_row() {
    let _box = BOX.lock().unwrap_or_else(|e| e.into_inner());
    let n = 3usize;
    assert!(cores() >= 2 * n, "needs a box with >= {} cores", 2 * n);

    let burners = Burners::spawn(n);
    std::thread::sleep(Duration::from_millis(500));
    let census = CpuCensus::start(&[std::process::id()]).expect("the ps path must be available");
    std::thread::sleep(Duration::from_secs(3));
    let ext = census.finish().expect("a census");
    drop(burners);

    // Two thirds of the spawned load, to leave room for a scheduler that does
    // not give three spinners three full cores on a box with a desktop on it.
    let floor = n as f64 * 2.0 / 3.0;
    assert!(
        ext.external_cores_during_row >= floor,
        "{n} burning processes outside the tree read as {:.2} cores; the census must see at \
         least {floor:.2} or it cannot detect the contention it exists to detect ({:.2}s \
         window, method {})",
        ext.external_cores_during_row,
        ext.window_s,
        ext.method
    );
    assert!(
        void_at_load(ext.external_cores_during_row, LOAD_VOID_ABOVE_SMALL_HOST),
        "{:.2} external cores must void at andromeda's threshold",
        ext.external_cores_during_row
    );
}

/// Spinning threads INSIDE the lane's own process: that is the lane running,
/// and it must not read as contention.
///
/// This is the half that matters most on this platform, because it is the half
/// `load1` gets wrong: four threads of our own work lift the Mach load average
/// past 2.0 by themselves, which is how the old rule voided rows for running.
#[test]
fn the_lanes_own_threads_do_not_void_the_row() {
    let _box = BOX.lock().unwrap_or_else(|e| e.into_inner());
    let n = 4usize;
    assert!(cores() >= 2 * n, "needs a box with >= {} cores", 2 * n);

    let stop = Arc::new(AtomicBool::new(false));
    let spinners: Vec<_> = (0..n)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut x = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                }
                x
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(500));
    let c = own_contribution(&[std::process::id()], || {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    stop.store(true, Ordering::Relaxed);
    for s in spinners {
        let _ = s.join();
    }
    assert_attributed_to_the_lane("4 spinning threads inside the lane", n, c);
}
