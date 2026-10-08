// SPDX-License-Identifier: Apache-2.0
//! E7's A10 test: the results doc cannot cite a row that is not published.
//!
//! Prose outlives the run that produced it. `check_qec.py` was advertised as
//! using qsim as an extra oracle and never imported it — "advertised and
//! absent" — and a results doc is where that rots fastest. So:
//!
//! 1. every row id cited in `docs/EMULATOR-COMPARISON.md` (a backticked token
//!    whose first segment is a lane prefix) must exist in a lane JSONL;
//! 2. a cited speed or capability row must deserialize through the schema
//!    (every witness key present), re-serialize (every executed-path witness
//!    asserted, nothing printed-only), and pass [`check`] — a cited VoidRow
//!    must exist and parse as one;
//! 3. every published speed or capability row must be cited at least once,
//!    so an extractor that silently finds nothing cannot pass, and a row
//!    cannot be published and left out of the doc.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use omega_emu_compare::{check, Row, VoidRow};

/// First segments of every lane's row ids. A citation is recognised by these;
/// a JSONL row whose prefix is missing here fails the test, so the list
/// cannot fall behind the lanes.
const PREFIXES: &[&str] = &[
    "stab", "stabrank", "stim", "dense", "gpu", "ferm", "mps", "pp", "cv",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Lane JSONL files: any `.jsonl` under `tools/` or `crates/` whose first line
/// is a row or void (has `row_id` and `lane`). Oracle fixtures are not rows.
fn lane_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if name != "target" && !name.starts_with('.') && name != "node_modules" {
                lane_files(&p, out);
            }
        } else if name.ends_with(".jsonl") {
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            if let Some(first) = text.lines().next() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(first) {
                    if v.get("row_id").is_some() && v.get("lane").is_some() {
                        out.push(p);
                    }
                }
            }
        }
    }
}

pub struct Published {
    /// row id -> (file, raw line, is_void)
    pub rows: BTreeMap<String, (PathBuf, String, bool)>,
}

pub fn published() -> Published {
    let root = root();
    let mut files = Vec::new();
    lane_files(&root.join("tools"), &mut files);
    lane_files(&root.join("crates"), &mut files);
    let mut rows = BTreeMap::new();
    for f in files {
        for line in std::fs::read_to_string(&f).unwrap().lines() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let id = v["row_id"].as_str().unwrap().to_string();
            let void = v.get("reason").is_some() && v.get("body").is_none();
            // A later line for the same id (a re-run appended) supersedes.
            rows.insert(id, (f.clone(), line.to_string(), void));
        }
    }
    Published { rows }
}

pub fn citations(doc: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for chunk in doc.split('`').skip(1).step_by(2) {
        let t = chunk.trim();
        let ok_chars = t
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.');
        if !ok_chars || !t.contains('-') {
            continue;
        }
        let first = t.split('-').next().unwrap_or("");
        if PREFIXES.contains(&first) {
            out.insert(t.to_string());
        }
    }
    out
}

/// Every reason the doc and the rows disagree. Empty means the doc is honest
/// about what is published.
pub fn audit(doc: &str, published: &Published) -> Vec<String> {
    let mut bad = Vec::new();
    let cited = citations(doc);
    for id in published.rows.keys() {
        let first = id.split('-').next().unwrap_or("");
        if !PREFIXES.contains(&first) {
            bad.push(format!(
                "row {id} has prefix {first:?}, unknown to the citation extractor"
            ));
        }
    }
    for id in &cited {
        match published.rows.get(id) {
            None => bad.push(format!(
                "the doc cites `{id}`, which no lane JSONL publishes"
            )),
            Some((file, line, true)) => {
                if let Err(e) = serde_json::from_str::<VoidRow>(line) {
                    bad.push(format!(
                        "`{id}` in {} is not a well-formed VoidRow: {e}",
                        file.display()
                    ));
                }
            }
            Some((file, line, false)) => match serde_json::from_str::<Row>(line) {
                Err(e) => bad.push(format!(
                    "`{id}` in {} does not deserialize (a witness key is missing?): {e}",
                    file.display()
                )),
                Ok(row) => {
                    if let Err(e) = serde_json::to_string(&row) {
                        bad.push(format!("`{id}` does not re-serialize (a witness is printed, not asserted?): {e}"));
                    }
                    let problems = check(&row);
                    if !problems.is_empty() {
                        bad.push(format!("`{id}` fails check(): {problems:?}"));
                    }
                }
            },
        }
    }
    for (id, (file, _, void)) in &published.rows {
        if !void && !cited.contains(id) {
            bad.push(format!(
                "{} publishes `{id}` but the doc never cites it",
                file.display()
            ));
        }
    }
    bad
}

#[test]
fn every_row_the_doc_cites_is_published_with_its_witnesses() {
    let doc = std::fs::read_to_string(root().join("docs/EMULATOR-COMPARISON.md")).unwrap();
    let published = published();
    // 35 = the speed and capability rows published at E7's close (E2 8, E3 1,
    // E4 11, E6 6, E7 9). It only rises.
    let speed_or_cap = published.rows.values().filter(|(_, _, v)| !v).count();
    assert!(
        speed_or_cap >= 41,
        "found only {speed_or_cap} published rows: the lane scan is vacuous"
    );
    assert!(
        citations(&doc).len() >= speed_or_cap,
        "fewer citations than published rows"
    );
    let bad = audit(&doc, &published);
    assert!(
        bad.is_empty(),
        "the doc and the rows disagree:\n{}",
        bad.join("\n")
    );
}

#[test]
fn a_citation_of_an_unpublished_row_is_refused() {
    let doc = std::fs::read_to_string(root().join("docs/EMULATOR-COMPARISON.md")).unwrap();
    let mut forged = doc.clone();
    forged.push_str("\nAnd `dense-hea-99q-d4-o-expectation-vs-qulacs` was a win.\n");
    let bad = audit(&forged, &published());
    assert!(
        bad.iter()
            .any(|b| b.contains("dense-hea-99q-d4") && b.contains("no lane JSONL")),
        "{bad:?}"
    );
}

#[test]
fn a_cited_row_with_a_witness_removed_is_refused() {
    let doc = std::fs::read_to_string(root().join("docs/EMULATOR-COMPARISON.md")).unwrap();
    let mut p = published();
    let id = "dense-random1-24q-d20-o-expectation-vs-qulacs";
    let (f, line, void) = p.rows.get(id).cloned().expect("the row exists");
    let mut v: serde_json::Value = serde_json::from_str(&line).unwrap();
    v["body"]["speed"]["value_gate"].take();
    v["body"]["speed"]
        .as_object_mut()
        .unwrap()
        .remove("value_gate");
    p.rows
        .insert(id.to_string(), (f.clone(), v.to_string(), void));
    let bad = audit(&doc, &p);
    assert!(
        bad.iter()
            .any(|b| b.contains(id) && b.contains("does not deserialize")),
        "{bad:?}"
    );
    // And a printed-only executed-path witness.
    let mut v: serde_json::Value = serde_json::from_str(&line).unwrap();
    v["body"]["speed"]["ours"]["arm"]["executed_path"][0]["asserted"] = serde_json::json!(false);
    p.rows.insert(id.to_string(), (f, v.to_string(), void));
    let bad = audit(&doc, &p);
    assert!(
        bad.iter()
            .any(|b| b.contains(id) && b.contains("re-serialize")),
        "{bad:?}"
    );
}

#[test]
fn an_uncited_published_row_is_refused() {
    let doc = std::fs::read_to_string(root().join("docs/EMULATOR-COMPARISON.md")).unwrap();
    let id = "ferm-lucj16";
    let scrubbed = doc.replace(&format!("`{id}`"), "the LUCJ row");
    let bad = audit(&scrubbed, &published());
    assert!(
        bad.iter()
            .any(|b| b.contains(id) && b.contains("never cites")),
        "{bad:?}"
    );
}
