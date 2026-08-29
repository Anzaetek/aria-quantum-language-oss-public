// SPDX-License-Identifier: Apache-2.0
//! The shared file, and the three rules that keep it from lying.
//!
//! # 1. The write is atomic, or the file is a liability
//!
//! The obvious implementation — seek to zero, truncate, write — has a window
//! between the truncate and the write in which the file is **empty**. A process
//! that dies in that window (and processes dying is the entire premise of this
//! crate) leaves a zero-length file behind.
//!
//! That is not a small corruption. An empty ledger is indistinguishable from an
//! empty *gate*: every holder's record is gone while every holder's memory is
//! still resident, so the next request is admitted on top of all of it. The
//! model in `proofs/tla/HostGate.tla` reaches that state in **two steps** and it
//! violates `AccountingCoversReality`.
//!
//! So: write a temp file in the same directory, `fsync` it, `rename` over the
//! target, then `fsync` the directory. A crash at any point leaves either the
//! old ledger or the new one, and never a torn one.
//!
//! # 2. The lock is on a file that is never renamed
//!
//! `flock` attaches to an open file description, not to a path. Lock the ledger
//! itself and then rename over it — which rule 1 requires — and the next process
//! opens a *different inode*, locks that, and now two processes each hold "the"
//! exclusive lock on two different files. Mutual exclusion is gone with no error
//! anywhere.
//!
//! The same split happens the moment anyone moves a corrupt ledger aside, which
//! is exactly what a naive recovery path does. So the lock lives on its own
//! file, which is created once and never renamed, never unlinked, and never
//! written to.
//!
//! # 3. An unreadable ledger is an error, never an empty one
//!
//! Zero-length, truncated, wrong schema: every one of them returns an error and
//! leaves the file alone. The one thing this module must never do is treat "I
//! could not read the budget" as "there is no budget", because that is the same
//! failure as rule 1 arriving by a different road.
//!
//! Unknown *fields*, by contrast, are preserved and ignored. Three separate
//! repositories deploy processes against one ledger on their own schedules, so
//! version skew is the normal state and not an incident. A reader that refused
//! unknown fields would turn a routine upgrade of one consumer into an outage
//! for the other two.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::identity::HolderId;
use crate::{Amounts, Axis};

/// Bumped only for a change an older reader cannot safely ignore. Adding a
/// field is not such a change — see rule 3.
pub const SCHEMA: &str = "omega-hostgate-1";

/// One holder's claim on the budget.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Record {
    pub pid: u32,
    /// The incarnation stamp. Zero means "not recorded", which is treated as
    /// unverifiable rather than as a match.
    #[serde(default)]
    pub start: u64,
    #[serde(default)]
    pub tag: String,
    /// Axes are stored as their display names so the file stays readable by a
    /// human and by a Python twin that does not share our enum.
    pub amounts: BTreeMap<String, u64>,
    /// Anything a newer writer added. Preserved on rewrite so an older process
    /// does not silently delete a field it did not understand.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Record {
    pub fn holder(&self) -> HolderId {
        HolderId {
            pid: self.pid,
            start: self.start,
        }
    }

    pub fn amount(&self, axis: &Axis) -> u64 {
        self.amounts.get(&axis.to_string()).copied().unwrap_or(0)
    }
}

/// The whole file.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Ledger {
    pub schema: String,
    /// Which machine lifetime this file belongs to — in practice
    /// [`crate::config::boot_id`].
    ///
    /// A ledger that outlives a reboot describes holders that cannot exist, and
    /// on Linux `(pid, start_time)` does not catch that: field 22 of
    /// `/proc/<pid>/stat` counts ticks since boot, so the same pid and the same
    /// tick count recur across boots. A mismatch here drops every holder, which
    /// is safe precisely because the file parsed correctly and told us so — this
    /// is not the "unreadable means empty" mistake rule 3 forbids.
    ///
    /// Never a clock. macOS's `kern.boottime` is recomputed as now-minus-uptime
    /// and moves on every NTP step and sleep/wake, which would invalidate every
    /// live grant on a laptop several times a day.
    pub instance: String,
    pub caps: BTreeMap<String, u64>,
    #[serde(default)]
    pub holders: Vec<Record>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Ledger {
    pub fn new(instance: impl Into<String>, caps: &Amounts) -> Self {
        Ledger {
            schema: SCHEMA.to_string(),
            instance: instance.into(),
            caps: caps.iter().map(|(a, v)| (a.to_string(), *v)).collect(),
            holders: Vec::new(),
            extra: BTreeMap::new(),
        }
    }

    pub fn cap(&self, axis: &Axis) -> u64 {
        self.caps.get(&axis.to_string()).copied().unwrap_or(0)
    }

    /// Total currently charged on one axis.
    pub fn charged(&self, axis: &Axis) -> u64 {
        self.holders.iter().map(|r| r.amount(axis)).sum()
    }
}

/// What went wrong reaching the file. Every variant is a refusal, never a
/// silently empty budget.
#[derive(Debug)]
pub enum LedgerError {
    Io(std::io::Error),
    /// Zero-length, truncated, or not JSON. Rule 3: this is an error.
    Corrupt(String),
    /// A schema this reader cannot safely participate in. The file is left
    /// untouched — an older process bows out rather than rewriting a newer
    /// format it would damage.
    Schema {
        found: String,
        expected: String,
    },
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::Io(e) => write!(f, "{e}"),
            LedgerError::Corrupt(why) => write!(
                f,
                "the ledger is unreadable ({why}) — refusing rather than \
                 treating an unreadable budget as an empty one"
            ),
            LedgerError::Schema { found, expected } => write!(
                f,
                "ledger schema is {found:?}, this build speaks {expected:?} — \
                 leaving the file alone rather than rewriting a format it may \
                 not understand"
            ),
        }
    }
}

impl From<std::io::Error> for LedgerError {
    fn from(e: std::io::Error) -> Self {
        LedgerError::Io(e)
    }
}

/// The ledger file plus its lock file.
pub struct Store {
    path: PathBuf,
    lock_path: PathBuf,
}

impl Store {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        // Never renamed, never unlinked — see rule 2.
        let lock_path = path.with_extension("lock");
        Store { path, lock_path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Take the lock, hand the caller the current ledger, and write it back only
    /// if the caller says it changed.
    ///
    /// The closure returns `(value, dirty)`. Separating those two is what lets
    /// an inspection prune, report what it found, and still not rewrite a file
    /// it did not change — a `/health` endpoint polled every few seconds should
    /// not be a write load on a file every acquire on the box must lock.
    pub fn with_lock<T>(
        &self,
        instance: &str,
        caps: &Amounts,
        f: impl FnOnce(&mut Ledger) -> (T, bool),
    ) -> Result<T, LedgerError> {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.lock_path)?;
        lock.lock()?;
        let out = (|| {
            let (mut ledger, stale) = match self.read()? {
                // A ledger stamped with a different boot describes processes
                // that cannot exist. This is NOT the "unreadable means empty"
                // mistake — the file parsed perfectly and says, in its own
                // words, that it belongs to a machine lifetime that has ended.
                Some(l) if l.instance != instance => {
                    let dropped = l.holders.len();
                    if dropped > 0 {
                        eprintln!(
                            "[hostgate] ledger is from a previous boot \
                             ({} != {}); dropping {dropped} stale holder(s)",
                            l.instance, instance
                        );
                    }
                    (Ledger::new(instance, caps), true)
                }
                Some(l) => (l, false),
                None => (Ledger::new(instance, caps), false),
            };
            let (value, dirty) = f(&mut ledger);
            if dirty || stale {
                self.write(&ledger)?;
            }
            Ok(value)
        })();
        // Released on drop too, but doing it explicitly keeps the window
        // between the write and the unlock as short as it reads.
        let _ = lock.unlock();
        out
    }

    /// `Ok(None)` means the file does not exist yet — the only benign absence.
    fn read(&self) -> Result<Option<Ledger>, LedgerError> {
        let mut file = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;
        if buf.trim().is_empty() {
            // Rule 3, and the single most important line in this file: an empty
            // ledger is NOT an empty gate.
            return Err(LedgerError::Corrupt(
                "the file is empty, which means a writer died mid-rewrite — \
                 an empty ledger is not an empty budget"
                    .into(),
            ));
        }
        let ledger: Ledger = serde_json::from_str(&buf)
            .map_err(|e| LedgerError::Corrupt(format!("not valid ledger JSON: {e}")))?;
        if ledger.schema != SCHEMA {
            return Err(LedgerError::Schema {
                found: ledger.schema,
                expected: SCHEMA.to_string(),
            });
        }
        Ok(Some(ledger))
    }

    /// Temp file, fsync, rename, fsync the directory. See rule 1.
    fn write(&self, ledger: &Ledger) -> Result<(), LedgerError> {
        let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
        let tmp = dir.join(format!(
            ".{}.{}.tmp",
            self.path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "hostgate".into()),
            std::process::id()
        ));
        let body = serde_json::to_vec_pretty(ledger)
            .map_err(|e| LedgerError::Corrupt(format!("cannot serialise: {e}")))?;
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&body)?;
            // Durable before it is visible: without this the rename can be
            // ordered ahead of the data on some filesystems, which puts a
            // zero-length file at the destination — the very state rule 1
            // exists to prevent.
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        // Make the rename itself durable. Best-effort: not every platform
        // permits opening a directory, and failing here would refuse a write
        // that has already succeeded.
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static N: AtomicU32 = AtomicU32::new(0);

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "omega-hostgate-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn caps() -> Amounts {
        let mut c = Amounts::new();
        c.insert(Axis::HostBytes, 1 << 30);
        c.insert(Axis::Slots, 8);
        c
    }

    fn rec(pid: u32, start: u64, bytes: u64) -> Record {
        let mut amounts = BTreeMap::new();
        amounts.insert(Axis::HostBytes.to_string(), bytes);
        Record {
            pid,
            start,
            tag: "t".into(),
            amounts,
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn a_missing_ledger_is_created_and_round_trips() {
        let d = tmpdir();
        let s = Store::new(d.join("gate.json"));
        s.with_lock("inst-1", &caps(), |l| {
            l.holders.push(rec(42, 7, 1 << 20));
            ((), true)
        })
        .unwrap();

        let seen = s
            .with_lock("inst-1", &caps(), |l| (l.clone(), false))
            .unwrap();
        assert_eq!(seen.instance, "inst-1");
        assert_eq!(seen.holders.len(), 1);
        assert_eq!(seen.holders[0].holder(), HolderId { pid: 42, start: 7 });
        assert_eq!(seen.charged(&Axis::HostBytes), 1 << 20);
        assert_eq!(seen.cap(&Axis::Slots), 8);
    }

    #[test]
    fn an_empty_file_is_an_error_not_an_empty_budget() {
        // The two-step crash from the model: a writer died mid-rewrite.
        let d = tmpdir();
        let p = d.join("gate.json");
        std::fs::write(&p, "").unwrap();
        let s = Store::new(&p);
        let err = s
            .with_lock("inst-1", &caps(), |_l| ((), true))
            .expect_err("an empty ledger must refuse");
        match err {
            LedgerError::Corrupt(why) => {
                assert!(why.contains("empty"), "{why}");
            }
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    #[test]
    fn a_truncated_file_is_an_error_and_is_left_alone() {
        let d = tmpdir();
        let p = d.join("gate.json");
        std::fs::write(&p, "{\"schema\":\"omega-hostgate-1\",\"holders\":[").unwrap();
        let s = Store::new(&p);
        assert!(s.with_lock("i", &caps(), |_| ((), true)).is_err());
        // Untouched: moving it aside would split the lock domain.
        assert!(std::fs::read_to_string(&p).unwrap().starts_with('{'));
    }

    #[test]
    fn a_newer_schema_makes_this_reader_bow_out_without_rewriting() {
        let d = tmpdir();
        let p = d.join("gate.json");
        std::fs::write(
            &p,
            r#"{"schema":"omega-hostgate-9","instance":"i","caps":{},"holders":[]}"#,
        )
        .unwrap();
        let before = std::fs::read_to_string(&p).unwrap();
        let s = Store::new(&p);
        match s.with_lock("i", &caps(), |_| ((), true)) {
            Err(LedgerError::Schema { found, .. }) => assert_eq!(found, "omega-hostgate-9"),
            other => panic!("expected a schema refusal, got {other:?}"),
        }
        assert_eq!(std::fs::read_to_string(&p).unwrap(), before);
    }

    #[test]
    fn unknown_fields_survive_a_rewrite_by_an_older_reader() {
        // Three repos upgrade on their own schedules; a reader that dropped
        // what it did not understand would corrupt a newer writer's state.
        let d = tmpdir();
        let p = d.join("gate.json");
        std::fs::write(
            &p,
            r#"{"schema":"omega-hostgate-1","instance":"i","caps":{"slots":4},
                "holders":[{"pid":1,"start":2,"amounts":{"slots":1},
                            "covered_by":"grant-xyz"}],
                "future_top_level":{"k":1}}"#,
        )
        .unwrap();
        let s = Store::new(&p);
        s.with_lock("i", &caps(), |l| {
            l.holders.push(rec(9, 9, 1));
            ((), true)
        })
        .unwrap();

        let back: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(
            back["future_top_level"]["k"], 1,
            "top-level extra must survive"
        );
        assert_eq!(
            back["holders"][0]["covered_by"], "grant-xyz",
            "a record's unknown fields must survive"
        );
    }

    #[test]
    fn a_ledger_from_a_previous_boot_drops_its_holders() {
        // On Linux (pid, start_time) can alias across a reboot, because the
        // start time counts ticks since boot. The boot stamp is what actually
        // closes it, and this is the case that would otherwise charge a live
        // unrelated process forever.
        let d = tmpdir();
        let p = d.join("gate.json");
        let s = Store::new(&p);
        s.with_lock("boot-A", &caps(), |l| {
            l.holders.push(rec(42, 7, 1 << 20));
            l.holders.push(rec(43, 9, 1 << 20));
            ((), true)
        })
        .unwrap();

        let seen = s
            .with_lock("boot-B", &caps(), |l| (l.clone(), false))
            .unwrap();
        assert!(
            seen.holders.is_empty(),
            "holders from a previous boot cannot exist and must not be charged"
        );
        assert_eq!(seen.instance, "boot-B");

        // And the reset is persisted, not merely returned.
        let again = s
            .with_lock("boot-B", &caps(), |l| (l.clone(), false))
            .unwrap();
        assert!(again.holders.is_empty());
        assert_eq!(again.instance, "boot-B");
    }

    #[test]
    fn the_same_boot_keeps_its_holders() {
        let d = tmpdir();
        let p = d.join("gate.json");
        let s = Store::new(&p);
        s.with_lock("boot-A", &caps(), |l| {
            l.holders.push(rec(42, 7, 1 << 20));
            ((), true)
        })
        .unwrap();
        let seen = s
            .with_lock("boot-A", &caps(), |l| (l.clone(), false))
            .unwrap();
        assert_eq!(seen.holders.len(), 1);
    }

    #[test]
    fn returning_none_does_not_rewrite_the_file() {
        let d = tmpdir();
        let p = d.join("gate.json");
        let s = Store::new(&p);
        s.with_lock("i", &caps(), |l| {
            l.holders.push(rec(1, 1, 5));
            ((), true)
        })
        .unwrap();
        let before = std::fs::metadata(&p).unwrap().len();
        s.with_lock("i", &caps(), |l| {
            assert_eq!(l.holders.len(), 1);
            ((), false)
        })
        .unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().len(), before);
    }

    #[test]
    fn the_lock_is_a_separate_file_that_the_rename_cannot_replace() {
        let d = tmpdir();
        let p = d.join("gate.json");
        let s = Store::new(&p);
        s.with_lock("i", &caps(), |_| ((), true)).unwrap();
        let lock = d.join("gate.lock");
        assert!(
            lock.exists(),
            "the lock file must exist alongside the ledger"
        );
        let ino_before = std::fs::metadata(&lock).unwrap();
        // A second transaction renames the LEDGER; the lock file must be the
        // same inode afterwards or mutual exclusion has quietly split.
        s.with_lock("i", &caps(), |l| {
            l.holders.push(rec(2, 2, 1));
            ((), true)
        })
        .unwrap();
        let ino_after = std::fs::metadata(&lock).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(ino_before.ino(), ino_after.ino());
        }
        let _ = ino_before;
        let _ = ino_after;
    }

    #[test]
    fn no_temp_file_is_left_behind() {
        let d = tmpdir();
        let s = Store::new(d.join("gate.json"));
        s.with_lock("i", &caps(), |l| {
            l.holders.push(rec(1, 1, 1));
            ((), true)
        })
        .unwrap();
        let strays: Vec<_> = std::fs::read_dir(&d)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(strays.is_empty(), "left temp files behind: {strays:?}");
    }
}
