// SPDX-License-Identifier: Apache-2.0
//! §4.4a's census, proven to fire. Replacing a gate that fires too often with
//! one that never fires is the cheapest way to make the comparison worthless,
//! so these tests burn real cores: work OUTSIDE the lane's tree must void, and
//! work INSIDE it must not. Each one fails if the census counts the wrong tree.
#![cfg(target_os = "linux")]

use omega_emu_compare::{external_cpu_since, void_at_load, CpuCensus, LOAD_VOID_ABOVE_LARGE_HOST};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The two burning tests would read each other's work as external, so they
/// never overlap, whatever the harness's thread count.
static BOX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// The two "own work must not void" tests share the box with whatever else is
/// running, and a neighbour's burst is real external CPU that this census is
/// right to count. A census that leaked the lane's own work would read about
/// twelve to sixteen extra cores on EVERY attempt, so each of those tests takes
/// up to five attempts and asserts on the quietest one: a leak still fails, a
/// transient neighbour does not.
const OWN_WORK_ATTEMPTS: usize = 5;

fn quietest_of(mut attempt: impl FnMut() -> f64, below: f64) -> (f64, Vec<f64>) {
    let mut seen = Vec::new();
    for _ in 0..OWN_WORK_ATTEMPTS {
        seen.push(attempt());
        if *seen.last().unwrap() < below {
            break;
        }
    }
    (seen.iter().cloned().fold(f64::INFINITY, f64::min), seen)
}

#[test]
fn the_arithmetic_subtracts_the_own_tree() {
    // 30 s of system busy time, 26 s of it ours, over 2 s of wall at HZ 100.
    let e = external_cpu_since(3000, 2600, 100.0, 2.0, &[1]);
    assert!((e.external_cores_during_row - 2.0).abs() < 1e-12, "{e:?}");
}

/// Twelve busy processes that are NOT in the lane's tree: the census must see
/// about twelve cores and void the row at akilles' threshold of 8.
#[test]
fn someone_elses_work_voids_the_row() {
    let _box = BOX.lock().unwrap_or_else(|e| e.into_inner());
    let n = 12usize;
    assert!(cores() >= 2 * n, "needs a box with >= {} cores", 2 * n);
    let mut burners: Vec<Child> = (0..n)
        .map(|_| {
            Command::new("sh")
                .arg("-c")
                .arg("while :; do :; done")
                .spawn()
                .unwrap()
        })
        .collect();
    std::thread::sleep(Duration::from_millis(300));
    let census = CpuCensus::start(&[std::process::id()]).unwrap();
    std::thread::sleep(Duration::from_secs(3));
    let ext = census.finish().unwrap();
    for b in &mut burners {
        let _ = b.kill();
        let _ = b.wait();
    }
    assert!(
        ext.external_cores_during_row > 10.0,
        "12 burning processes outside the tree read as {:.2} cores",
        ext.external_cores_during_row
    );
    assert!(void_at_load(
        ext.external_cores_during_row,
        LOAD_VOID_ABOVE_LARGE_HOST
    ));
}

/// Sixteen spinning threads INSIDE the lane's process: that is the lane
/// running, and it must not read as contention.
#[test]
fn the_lanes_own_threads_do_not_void_the_row() {
    let _box = BOX.lock().unwrap_or_else(|e| e.into_inner());
    let n = 16usize;
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
    std::thread::sleep(Duration::from_millis(300));
    let (quietest, seen) = quietest_of(
        || {
            let census = CpuCensus::start(&[std::process::id()]).unwrap();
            let t = Instant::now();
            while t.elapsed() < Duration::from_secs(3) {
                std::thread::sleep(Duration::from_millis(50));
            }
            census.finish().unwrap().external_cores_during_row
        },
        4.0,
    );
    stop.store(true, Ordering::Relaxed);
    for s in spinners {
        let _ = s.join();
    }
    assert!(
        quietest < 4.0,
        "16 of the lane's own threads read as {quietest:.2} external cores at best (attempts: {seen:.2?})"
    );
}

/// Twelve busy CHILDREN of the lane that it starts and reaps INSIDE the row:
/// that is the lane running (the GPU lane starts one arm process per repeat,
/// so a GPU process never coexists with another), and it must not read as
/// contention. Their time reaches the lane through cutime/cstime once they
/// are waited for, which is why the burners here are reaped before `finish`.
#[test]
fn children_the_lane_reaps_during_the_row_are_its_own() {
    let _box = BOX.lock().unwrap_or_else(|e| e.into_inner());
    let n = 12usize;
    assert!(cores() >= 2 * n, "needs a box with >= {} cores", 2 * n);
    let (quietest, seen) = quietest_of(
        || {
            let census = CpuCensus::start(&[std::process::id()]).unwrap();
            let mut burners: Vec<Child> = (0..n)
                .map(|_| {
                    Command::new("sh")
                        .arg("-c")
                        .arg("while :; do :; done")
                        .spawn()
                        .unwrap()
                })
                .collect();
            std::thread::sleep(Duration::from_secs(3));
            for b in &mut burners {
                let _ = b.kill();
                let _ = b.wait();
            }
            census.finish().unwrap().external_cores_during_row
        },
        4.0,
    );
    assert!(
        quietest < 4.0,
        "12 of the lane's own reaped children read as {quietest:.2} external cores at best (attempts: {seen:.2?})"
    );
}

/// The unsafe direction. A child of the lane that burned CPU BEFORE the census
/// opened and is reaped INSIDE it credits all of that CPU to the lane's own
/// cutime. Without correction, `external = all − own` goes negative, clamps to
/// zero and reads as a quiet box while burners outside the tree are running.
#[test]
fn a_child_reaped_inside_the_window_does_not_hide_external_work() {
    let _box = BOX.lock().unwrap_or_else(|e| e.into_inner());
    assert!(cores() >= 24, "needs a box with >= 24 cores");
    // The lane's child: 8 busy processes under one `sh`, ~3 s before the
    // census, then it sits until we reap it inside the window.
    let mut child = Command::new("sh")
        .arg("-c")
        .arg("for i in 1 2 3 4 5 6 7 8; do (end=$(( $(date +%s) + 3 )); while [ $(date +%s) -lt $end ]; do :; done) & done; wait")
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(3500));
    // Six burners OUTSIDE the tree (children of a separate `sh` we do not
    // declare and do not reap in the window).
    let mut outside = Command::new("sh")
        .arg("-c")
        .arg("for i in 1 2 3 4 5 6; do (while :; do :; done) & done; wait")
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let census = CpuCensus::start(&[std::process::id()]).unwrap();
    let _ = child.wait(); // reaped inside the window: ~24 core-seconds of cutime
    std::thread::sleep(Duration::from_secs(3));
    let ext = census.finish().unwrap();
    let _ = Command::new("pkill")
        .arg("-P")
        .arg(outside.id().to_string())
        .status();
    let _ = outside.kill();
    let _ = outside.wait();
    assert!(
        ext.external_cores_during_row > 4.0,
        "six burners outside the tree read as {:.2} cores once a pre-window child was reaped inside it",
        ext.external_cores_during_row
    );
}
