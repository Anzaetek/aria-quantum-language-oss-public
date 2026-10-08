// SPDX-License-Identifier: Apache-2.0
//! `omega-hostgate run --watch` against real processes.
//!
//! The forking test is the one that matters. A watchdog on the root PID
//! reported 17 MB while a child held 3.7 GiB, and a tnsim run under a 40G
//! declaration reached ~49 GB. Both are the same shape: the memory is not in
//! the process that was spawned, it is in something that process forked.
//! Measuring only the root must fail this test. The allocations stay in the
//! hundreds of megabytes, under the cap of this 16 GB machine.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const FIXTURE_C: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void touch(size_t bytes) {
    unsigned char *block = malloc(bytes);
    if (!block) {
        perror("malloc");
        _exit(1);
    }
    /* Every byte, and not a repeated one. macOS compresses a sparse or
       uniform page out of the resident set, and a watch that then sees a
       few megabytes has not been shown to fire. */
    volatile unsigned char *p = block;
    size_t i = 0;
    uint32_t x = 0xA5A5F00Du;
    while (i < bytes) {
        x = x * 1664525u + 1013904223u;
        p[i] = (unsigned char)(x >> 24);
        i++;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    if (strcmp(argv[1], "alloc") == 0) {
        size_t mb = (size_t)strtoul(argv[2], 0, 10);
        unsigned secs = (unsigned)strtoul(argv[3], 0, 10);
        touch(mb * 1024ul * 1024ul);
        sleep(secs);
        return 0;
    }
    if (strcmp(argv[1], "fork-alloc") == 0) {
        unsigned secs = (unsigned)strtoul(argv[2], 0, 10);
        size_t mb = (size_t)strtoul(argv[3], 0, 10);
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            char mb_s[32], sec_s[32];
            snprintf(mb_s, sizeof mb_s, "%zu", mb);
            snprintf(sec_s, sizeof sec_s, "%u", secs);
            execl(argv[0], argv[0], "alloc", mb_s, sec_s, (char *)0);
            perror("execl");
            _exit(127);
        }
        /* Parent stays small. The resident bytes are in the child. */
        int st = 0;
        if (waitpid(pid, &st, 0) < 0) return 1;
        return 0;
    }
    if (strcmp(argv[1], "small") == 0) {
        size_t mb = (size_t)strtoul(argv[2], 0, 10);
        unsigned secs = (unsigned)strtoul(argv[3], 0, 10);
        touch(mb * 1024ul * 1024ul);
        sleep(secs);
        return 0;
    }
    if (strcmp(argv[1], "exit") == 0) {
        return (int)strtoul(argv[2], 0, 10);
    }
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
"#;

fn fixture() -> &'static Path {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!(
            "omega-hostgate-watch-fixture-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let src = dir.join("fixture.c");
        let bin = dir.join("fixture");
        std::fs::write(&src, FIXTURE_C).unwrap();
        let out = Command::new("cc")
            .args(["-O2", "-o"])
            .arg(&bin)
            .arg(&src)
            .output()
            .expect("cc");
        assert!(
            out.status.success(),
            "cc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        bin
    })
}

fn hostgate(args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_omega-hostgate"));
    cmd.args(args);
    for key in [
        "OMEGA_HOSTGATE",
        "OMEGA_HOSTGATE_MODE",
        "OMEGA_HOSTGATE_PROFILE",
        "OMEGA_HOSTGATE_MAX_MEM",
        "OMEGA_HOSTGATE_MEM_FRACTION",
        "OMEGA_HOSTGATE_SLOTS",
        "OMEGA_HOSTGATE_CPU_FRACTION",
    ] {
        cmd.env_remove(key);
    }
    cmd
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn number_after(line: &str, key: &str) -> u64 {
    let i = line
        .find(key)
        .unwrap_or_else(|| panic!("no {key:?} in {line}"));
    let rest = &line[i + key.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("no integer after {key:?} in {line}"))
}

#[test]
fn a_forked_grandchild_over_the_cap_is_killed_and_the_exit_is_not_the_childs() {
    // 256 MiB in the grandchild, 64 MiB declared. The parent only wait()s.
    // Exit 5 is the watch, not the child's status (the child would exit 0
    // if it were allowed to finish, and a signal death is not 5 either —
    // the wrapper chooses 5 so a caller can tell them apart).
    let bin = fixture();
    let dir = std::env::temp_dir().join(format!(
        "omega-hostgate-watch-ledger-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ledger = dir.join("gate.json");

    let out = hostgate(&[
        "run",
        "--watch",
        "--watch-interval-ms",
        "50",
        "--host-bytes",
        "64M",
        "--",
    ])
    .arg(bin)
    .args(["fork-alloc", "6", "256"])
    .env("OMEGA_HOSTGATE", &ledger)
    .env("OMEGA_HOSTGATE_MAX_MEM", "8G")
    .output()
    .expect("run hostgate");

    let err = stderr_of(&out);
    assert_eq!(
        out.status.code(),
        Some(5),
        "the watch must exit 5, not the child's status.\nstderr:\n{err}"
    );
    assert!(
        err.contains("exceeded the declaration"),
        "the receipt has to say the cap fired:\n{err}"
    );
    assert!(
        err.contains("process tree killed"),
        "and that the tree was killed:\n{err}"
    );
    let line = err
        .lines()
        .find(|l| l.contains("omega-hostgate: watch: peak"))
        .unwrap_or_else(|| panic!("no peak line:\n{err}"));
    let peak = number_after(line, "peak ");
    let declared = number_after(line, "declared ");
    assert_eq!(declared, 64 * 1024 * 1024, "{line}");
    assert!(
        peak > declared,
        "the sample that fired has to be over the declaration: {line}"
    );

    let text = std::fs::read_to_string(&ledger).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        text.contains("\"holders\": []"),
        "the transferred grant must be released once the tree is dead, not left \
         charged until some later transaction:\n{text}"
    );
}

#[test]
fn a_run_inside_the_declaration_is_not_killed_and_the_receipt_names_both_numbers() {
    let bin = fixture();
    let out = hostgate(&[
        "run",
        "--watch",
        "--watch-interval-ms",
        "50",
        "--host-bytes",
        "256M",
        "--",
    ])
    .arg(bin)
    .args(["small", "8", "1"])
    .output()
    .expect("run hostgate");
    let err = stderr_of(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a run inside its declaration must keep the child's status.\n{err}"
    );
    let line = err
        .lines()
        .find(|l| l.contains("omega-hostgate: watch: peak"))
        .unwrap_or_else(|| panic!("no peak line:\n{err}"));
    assert!(line.contains("stayed inside the declaration"), "{line}");
    let peak = number_after(line, "peak ");
    let declared = number_after(line, "declared ");
    assert_eq!(declared, 256 * 1024 * 1024, "{line}");
    assert!(
        peak > 4 * 1024 * 1024,
        "8 MiB touched has to show up as more than a few pages: {line}"
    );
    assert!(peak < declared, "inside means under the cap: {line}");
}

#[test]
fn without_watch_the_childs_status_is_returned_and_nothing_is_said_about_a_watch() {
    let bin = fixture();
    let out = hostgate(&["run", "--host-bytes", "32M", "--"])
        .arg(bin)
        .args(["exit", "17"])
        .output()
        .expect("run hostgate");
    let err = stderr_of(&out);
    assert_eq!(
        out.status.code(),
        Some(17),
        "without --watch the wrapper returns the child's status.\n{err}"
    );
    assert!(
        !err.contains("omega-hostgate: watch:"),
        "no receipt, no census, no kill line:\n{err}"
    );
}

#[test]
fn without_watch_a_grandchild_over_the_declared_bytes_is_not_killed() {
    // The same shape as the kill test, and a declaration it does not fit in.
    // Absent the flag, that declaration is still only a ledger entry — the
    // grandchild runs and the wrapper exits 0. Turning the cap on by default
    // would make this exit 5.
    let bin = fixture();
    let out = hostgate(&["run", "--host-bytes", "32M", "--"])
        .arg(bin)
        .args(["fork-alloc", "1", "96"])
        .output()
        .expect("run hostgate");
    let err = stderr_of(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "without --watch an over-declaration grandchild is not killed.\n{err}"
    );
    assert!(
        !err.contains("omega-hostgate: watch:"),
        "and the watch does not speak:\n{err}"
    );
}
