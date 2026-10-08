// SPDX-License-Identifier: Apache-2.0
//! What happens when the configured budget and the ledger's disagree.
//!
//! The ledger records the caps it was created with. For the rest of the boot,
//! every later process read those and ignored its own — so on a box whose
//! ledger already existed, `OMEGA_HOSTGATE_MAX_MEM=8G` left the cap at 16 G and
//! `status` printed `env:OMEGA_HOSTGATE_MAX_MEM` in the "cap from" column
//! beside the 16 G it had not produced. An operator lowering a budget was told
//! it had worked.
//!
//! Two directions, and they pull against each other. A lowered cap must take
//! effect, or the knob is decoration. It must NOT take effect underneath a
//! grant already priced against the old number, or `charged` goes above `cap`
//! and the ledger cannot describe its own state. The rule is therefore
//! "adopt when nothing is held", and both halves are tested here — the second
//! one matters more, because it is the one whose failure hands a running
//! process's memory to the next request.

use std::collections::BTreeMap;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};

use omega_hostgate::gate::HostGate;
use omega_hostgate::identity::{self, Liveness};
use omega_hostgate::ledger::{Ledger, Record, Store};
use omega_hostgate::{Amounts, Axis, Mode, Request};

static N: AtomicU32 = AtomicU32::new(0);

fn workdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "omega-hostgate-cap-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn caps(host_bytes: u64) -> Amounts {
    let mut c = Amounts::new();
    c.insert(Axis::HostBytes, host_bytes);
    c
}

fn gate(path: &std::path::Path, host_bytes: u64) -> HostGate {
    HostGate::new(path, "boot-under-test", caps(host_bytes), Mode::Enforce)
}

fn report(g: &HostGate) -> (u64, u64, usize) {
    let r = g
        .capacity()
        .unwrap()
        .into_iter()
        .find(|r| r.axis == Axis::HostBytes)
        .expect("host_bytes is always an axis here");
    (r.cap, r.configured, r.holders)
}

fn request(bytes: u64, tag: &str) -> Request {
    Request::new(tag).want(Axis::HostBytes, bytes)
}

/// A child that will outlive the test unless something kills it.
fn spawn_sleeper() -> Child {
    Command::new("/bin/sh")
        .args(["-c", "sleep 300"])
        .spawn()
        .expect("could not spawn a child process")
}

/// Someone else's live grant, stamped with that process's real start time.
fn record_for(path: &std::path::Path, caps: &Amounts, pid: u32, amount: u64) {
    let start = match identity::probe(&[pid]).get(&pid).copied() {
        Some(Liveness::Alive { start }) => start,
        other => panic!("expected the child {pid} to be alive, saw {other:?}"),
    };
    Store::new(path)
        .with_lock("boot-under-test", caps, |ledger: &mut Ledger| {
            let mut amounts = BTreeMap::new();
            amounts.insert(Axis::HostBytes.to_string(), amount);
            ledger.holders.push(Record {
                pid,
                start,
                tag: format!("child-{pid}"),
                amounts,
                extra: BTreeMap::new(),
            });
            ((), true)
        })
        .expect("could not write the child's grant");
}

/// Force the ledger onto disk at a given cap.
///
/// Opening a gate and reading it does NOT do this: `capacity` writes only when
/// something changed, so a fresh gate that finds no file builds one in memory
/// and drops it. The first version of this test relied on that and was
/// therefore vacuous — it compared a brand-new 8 G ledger against itself, and
/// stayed green with adoption removed entirely. Caught by the mutation, not by
/// reading it.
fn persist_ledger_at(path: &std::path::Path, host_bytes: u64) {
    Store::new(path)
        .with_lock("boot-under-test", &caps(host_bytes), |_l: &mut Ledger| {
            ((), true)
        })
        .expect("could not create the ledger");
    // Read the file back as bytes rather than through the gate: the fixture
    // has to prove the ledger is on DISK at this cap, and asking the gate
    // would be asking the thing under test.
    let raw: String = std::fs::read_to_string(path)
        .expect("the ledger must exist on disk now")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    assert!(
        raw.contains(&format!("\"host_bytes\":{host_bytes}")),
        "the fixture itself must be what it claims, or every test below is \
         comparing a cap against itself; file said: {raw}"
    );
}

/// The knob works on an idle box. Without adoption the cap stays at 16 G, the
/// 12 G request is admitted, and this fails on the refusal that never came.
#[test]
fn a_lowered_cap_takes_effect_when_nothing_is_held() {
    let dir = workdir();
    let path = dir.join("ledger.json");

    // Create the ledger at the larger cap, the way a first run would — and
    // make sure it is actually on disk, which is the whole premise.
    persist_ledger_at(&path, 16 << 30);

    let small = gate(&path, 8 << 30);
    let (cap, configured, holders) = report(&small);
    assert_eq!(holders, 0, "nothing is held");
    assert_eq!(configured, 8 << 30);
    assert_eq!(
        cap,
        8 << 30,
        "an idle ledger must adopt the configured cap, or every budget knob is \
         inert for the rest of the boot"
    );

    let refusal = small
        .try_acquire(&request(12 << 30, "too-big-now"))
        .expect_err("12 G must not fit in an 8 G budget");
    let said = refusal.to_string();
    assert!(
        said.contains("8589934592"),
        "the refusal must name the cap actually in force, said: {said}"
    );
}

/// The dangerous direction. A live holder was priced against 16 G; adopting
/// 8 G underneath it would report `charged` above `cap`. The old cap stays,
/// and the divergence is visible in the report rather than inferred.
#[test]
fn a_lowered_cap_is_not_adopted_underneath_a_live_grant_and_the_report_says_so() {
    let dir = workdir();
    let path = dir.join("ledger.json");

    persist_ledger_at(&path, 16 << 30);

    let mut child = spawn_sleeper();
    record_for(&path, &caps(16 << 30), child.id(), 12 << 30);

    let small = gate(&path, 8 << 30);
    let (cap, configured, holders) = report(&small);
    assert_eq!(
        holders, 1,
        "the child's grant is live and must not be pruned"
    );
    assert_eq!(
        cap,
        16 << 30,
        "the cap a live grant was admitted under must stay in force"
    );
    assert_eq!(
        configured,
        8 << 30,
        "and the configured cap must still be reported, or the disagreement is \
         invisible to the operator who caused it"
    );
    assert_ne!(
        cap, configured,
        "this test is about the two disagreeing; if they agree it proves nothing"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// Once the holder is gone the prune runs first, so the adoption that could
/// not happen before happens now — without anyone re-running anything.
#[test]
fn the_cap_is_adopted_on_the_first_look_after_the_last_holder_dies() {
    let dir = workdir();
    let path = dir.join("ledger.json");

    persist_ledger_at(&path, 16 << 30);

    let mut child = spawn_sleeper();
    record_for(&path, &caps(16 << 30), child.id(), 12 << 30);

    let small = gate(&path, 8 << 30);
    assert_eq!(
        report(&small).0,
        16 << 30,
        "held open while the child lives"
    );

    let _ = child.kill();
    let _ = child.wait();

    let (cap, configured, holders) = report(&small);
    assert_eq!(holders, 0, "the dead child's grant is reclaimed");
    assert_eq!(cap, 8 << 30, "and the cap moves on that same look");
    assert_eq!(configured, 8 << 30);
}

/// Raising a budget is the same rule, and worth its own case: the asymmetry
/// people expect ("lowering is dangerous, raising is free") is not the rule
/// implemented, and a reader should see that stated rather than assume it.
#[test]
fn a_raised_cap_also_waits_for_the_holders_to_go() {
    let dir = workdir();
    let path = dir.join("ledger.json");

    persist_ledger_at(&path, 8 << 30);

    let mut child = spawn_sleeper();
    record_for(&path, &caps(8 << 30), child.id(), 4 << 30);

    let big = gate(&path, 16 << 30);
    let (cap, configured, _) = report(&big);
    assert_eq!(
        cap,
        8 << 30,
        "still the cap the live grant was admitted under"
    );
    assert_eq!(configured, 16 << 30);

    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(report(&big).0, 16 << 30);
}

/// The display, end to end. The bug this file exists for was visible exactly
/// here: `status` printed the ledger's 16 G next to the name of the variable
/// that asked for 8 G, so the one place an operator checks said the change had
/// landed. The number and its source have to agree, or the line must say they
/// do not.
///
/// The holder is made by the real binary rather than written by hand: a record
/// stamped with this test's own instance string looks like a previous boot to
/// the `status` process, which drops it — and then there is no holder, the cap
/// is adopted, and the test passes while testing nothing.
#[test]
fn status_never_attributes_a_cap_to_a_source_that_is_not_in_force() {
    let dir = workdir();
    let path = dir.join("ledger.json");

    let mut holder = std::process::Command::new(env!("CARGO_BIN_EXE_omega-hostgate"))
        .args([
            "run",
            "--host-bytes",
            "12G",
            "--",
            "/bin/sh",
            "-c",
            "sleep 30",
        ])
        .env("OMEGA_HOSTGATE", &path)
        .env("OMEGA_HOSTGATE_MODE", "enforce")
        .env("OMEGA_HOSTGATE_MAX_MEM", "16G")
        .env_remove("OMEGA_HOSTGATE_PROFILE")
        .env_remove("OMEGA_HOSTGATE_MEM_FRACTION")
        .spawn()
        .expect("could not start the holder");

    // Wait for the grant to be on disk rather than sleeping a guessed amount.
    let mut waited = 0;
    while !path.exists()
        || std::fs::read_to_string(&path)
            .unwrap_or_default()
            .contains("\"holders\": []")
    {
        std::thread::sleep(std::time::Duration::from_millis(50));
        waited += 50;
        assert!(waited < 10_000, "the holder never recorded its grant");
    }

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_omega-hostgate"))
        .arg("status")
        .env("OMEGA_HOSTGATE", &path)
        .env("OMEGA_HOSTGATE_MODE", "enforce")
        .env("OMEGA_HOSTGATE_MAX_MEM", "8G")
        .env_remove("OMEGA_HOSTGATE_PROFILE")
        .env_remove("OMEGA_HOSTGATE_MEM_FRACTION")
        .output()
        .expect("could not run omega-hostgate status");
    let said = String::from_utf8_lossy(&out.stdout).into_owned();

    let line = said
        .lines()
        .find(|l| l.starts_with("host_bytes"))
        .unwrap_or_else(|| panic!("no host_bytes row in:\n{said}"))
        .to_string();

    let _ = holder.kill();
    let _ = holder.wait();

    assert!(
        line.contains("16.0 G"),
        "the cap in force is the one the live grant was admitted under, said: {line}"
    );
    assert!(
        line.contains("not in force"),
        "the row must say the configured cap is not in force, said: {line}"
    );
    assert!(
        line.contains("8.0 G"),
        "and it must name the cap that was asked for, said: {line}"
    );
    // The precise failure: the variable's name sitting alone in the "cap from"
    // column, as though it had produced the 16 G beside it.
    assert!(
        !line.trim_end().ends_with("env:OMEGA_HOSTGATE_MAX_MEM"),
        "this is the original bug, verbatim: {line}"
    );
}
