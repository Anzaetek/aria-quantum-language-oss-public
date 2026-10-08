// SPDX-License-Identifier: Apache-2.0
//! §4.4's quiet-box threshold is per host. Both sides are pinned, so a
//! selection that collapses to one value — every host at 2.0, which makes the
//! 32-core replicate unrunnable, or every host at 8.0, which admits a
//! saturated 10-core box — turns this red.

use omega_emu_compare::{
    hostgate_cap_from, load_void_above, void_at_load, LOAD_VOID_ABOVE_LARGE_HOST,
    LOAD_VOID_ABOVE_SMALL_HOST, SMALL_HOST_MAX_CORES,
};

#[test]
fn a_small_host_gets_two_and_a_large_host_gets_eight() {
    // andromeda (M4, 10 cores) and akilles (32 threads), by name of the box.
    assert_eq!(load_void_above(10), 2.0);
    assert_eq!(load_void_above(32), 8.0);
    // The named constants are the values the doc's sentence rests on.
    assert_eq!(LOAD_VOID_ABOVE_SMALL_HOST, 2.0);
    assert_eq!(LOAD_VOID_ABOVE_LARGE_HOST, 8.0);
}

#[test]
fn the_boundary_sits_between_sixteen_and_seventeen_cores() {
    assert_eq!(SMALL_HOST_MAX_CORES, 16);
    assert_eq!(load_void_above(SMALL_HOST_MAX_CORES), 2.0);
    assert_eq!(load_void_above(SMALL_HOST_MAX_CORES + 1), 8.0);
    assert_eq!(load_void_above(1), 2.0);
}

#[test]
fn the_same_load_reads_differently_on_the_two_boxes() {
    assert!(
        void_at_load(6.0, load_void_above(10)),
        "6 on 10 cores is busy"
    );
    assert!(
        !void_at_load(6.0, load_void_above(32)),
        "6 on 32 cores is quiet"
    );
}

#[test]
fn the_threshold_itself_does_not_void() {
    for cores in [10, 32] {
        let t = load_void_above(cores);
        assert!(!void_at_load(t, t));
        assert!(void_at_load(t + 1e-9, t));
    }
}

/// Every `.rs` under `dir`, recursively — a helper module in a subdirectory
/// is still lane code.
fn rust_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// A float literal such as `2.0`, `8.`, `2.0_f64` or `2e0` anywhere on the
/// line. Not matched: format specs (`{:.2}`, `{:.12e}` — the digits follow a
/// dot), section references (`§4.4` — the digits follow a non-ASCII byte),
/// method calls (`v2.len()`), and identifiers (`load1_before`).
fn has_float_literal(line: &str) -> bool {
    let b = line.as_bytes();
    (0..b.len()).any(|i| {
        if !b[i].is_ascii_digit() {
            return false;
        }
        if i > 0 {
            let p = b[i - 1];
            if p.is_ascii_alphanumeric() || p == b'_' || p == b'.' || p >= 0x80 {
                return false;
            }
        }
        let mut k = i;
        while k < b.len() && (b[k].is_ascii_digit() || b[k] == b'_') {
            k += 1;
        }
        match (b.get(k), b.get(k + 1)) {
            (Some(b'.'), next) => !next.is_some_and(|c| c.is_ascii_alphabetic() || *c == b'.'),
            (Some(b'e' | b'E'), Some(c)) => c.is_ascii_digit() || *c == b'-' || *c == b'+',
            _ => false,
        }
    })
}

/// The duplicated-predicate guard: a TRIPWIRE, not a proof. Text cannot see
/// every spelling of a threshold (a constant named on one line and compared on
/// another passes), so the writer separately refuses any `void_above` that is
/// not one of §4.4's two values. What this does catch, each mutation-proven:
///
/// * a lane that never calls the shared functions (positive check per lane);
/// * a float literal on any non-comment line that mentions a load or the void
///   threshold, in either comparison order and any spelling (`2.0`, `8.0`,
///   `2.0_f64`, `2 .. < load`), in any file under the lane, recursively;
/// * a literal `hostgate_cap: … "--host-bytes …"`.
///
/// `#[cfg(test)]` modules are exempt (fixtures may name a load); the scan stops
/// at the first such line in a file, where this tree keeps them.
#[test]
fn no_lane_carries_its_own_load_threshold() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lanes = [
        "crates/omega-backend-mps/examples/mps_quimb_compare",
        "tools/emu_compare/fermionic/src",
        "tools/emu_compare/stim/src",
        "tools/emu_compare/dense/src",
        "tools/emu_compare/gpu/src",
        "tools/emu_compare/pauliprop/src",
    ];
    let mut scanned = 0;
    let mut offences = Vec::new();
    for lane in lanes {
        let mut files = Vec::new();
        rust_sources(&root.join(lane), &mut files);
        assert!(
            !files.is_empty(),
            "{lane}: no sources — the scan would be vacuous"
        );
        let mut lane_src = String::new();
        for path in files {
            scanned += 1;
            let src = std::fs::read_to_string(&path).unwrap();
            for (i, line) in src.lines().enumerate() {
                let t = line.trim();
                if t.starts_with("#[cfg(test)]") {
                    break;
                }
                lane_src.push_str(line);
                lane_src.push('\n');
                if t.starts_with("//") {
                    continue;
                }
                let lower = t.to_ascii_lowercase();
                let about_load = lower.contains("load")
                    || lower.contains("void_above")
                    || lower.contains("quiet");
                let literal_threshold = about_load && has_float_literal(t);
                let literal_cap = t.contains("\"--host-bytes");
                let literal_box = t.contains("box_name") && t.contains("Witness::present(\"");
                if literal_threshold || literal_cap || literal_box {
                    offences.push(format!("{}:{}: {t}", path.display(), i + 1));
                }
            }
        }
        for needed in ["host_load_void_above()", "host_cores()", "hostgate_cap()"] {
            let found = lane_src.contains(needed)
                || (needed == "host_load_void_above()"
                    && lane_src.contains("load_void_above(cores)"));
            if !found {
                offences.push(format!("{lane}: never calls {needed}"));
            }
        }
    }
    assert!(scanned >= 9, "scanned only {scanned} lane sources");
    assert!(
        offences.is_empty(),
        "a lane carries its own load threshold or hostgate cap instead of the shared one:\n{}",
        offences.join("\n")
    );
}

#[test]
fn the_float_literal_detector_is_not_itself_vacuous() {
    for yes in [
        "x > 2.0",
        "2.0 < load",
        "8.",
        "2.0_f64",
        "2e0",
        "(1.5)",
        "load <= 2.01",
    ] {
        assert!(has_float_literal(yes), "{yes}");
    }
    for no in [
        "{load:.2}",
        "{:.2}",
        "load_void_above(10)",
        "load1_before",
        "x.0",
        "v2.len()",
    ] {
        assert!(!has_float_literal(no), "{no}");
    }
}

#[test]
fn the_cap_is_the_launcher_value_and_absence_is_refused() {
    assert_eq!(hostgate_cap_from(Some("40G")).unwrap(), "--host-bytes 40G");
    assert_eq!(hostgate_cap_from(Some(" 2G ")).unwrap(), "--host-bytes 2G");
    for bad in [None, Some(""), Some("   "), Some("--host-bytes 2G")] {
        let err = hostgate_cap_from(bad).expect_err("no cap, no row");
        assert!(err.contains("OMEGA_EMU_HOSTGATE_CAP"), "{err}");
    }
}
