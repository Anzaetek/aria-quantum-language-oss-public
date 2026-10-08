// SPDX-License-Identifier: Apache-2.0
//! F4.6 — every fenced block in `docs/FERMIONICQASM.md` is exercised.
//!
//! A `fermionicqasm` block must parse and lower. A `fermionicqasm-refuse`
//! block must be refused, and the error must contain the needle written
//! after the tag. A `conventions-quote` block is not a program: each of its
//! lines has to occur in `conventions.json`, which is the F4.0 pin the
//! document is quoting. Any other fence fails. The document missing fails.
//!
//! Absent this test, a prose example can drift from the grammar and stay
//! green. Corrupt one block and this file reddens on that block.

use omega_core::circuit::CircuitType;
use omega_parser::lower_to_ir;
use std::path::PathBuf;

const ACCEPT: usize = 3;
const REFUSE: usize = 14;
const QUOTE: usize = 1;

fn repo_file(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read_doc() -> String {
    let path = repo_file("../../docs/FERMIONICQASM.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "docs/FERMIONICQASM.md is missing ({e} at {}). F4.6 has no language document.",
            path.display()
        )
    })
}

struct Block {
    /// 1-based line of the opening fence, for the failure message.
    line: usize,
    info: String,
    body: String,
}

fn fenced_blocks(md: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut open_at = None;
    let mut info = String::new();
    let mut body = String::new();
    for (idx, line) in md.lines().enumerate() {
        let line_no = idx + 1;
        if open_at.is_none() {
            if let Some(rest) = line.strip_prefix("```") {
                if rest.starts_with('`') {
                    continue;
                }
                open_at = Some(line_no);
                info = rest.trim().to_string();
                body.clear();
            }
        } else if line.trim() == "```" {
            out.push(Block {
                line: open_at.unwrap(),
                info: info.clone(),
                body: body.clone(),
            });
            open_at = None;
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    assert!(
        open_at.is_none(),
        "unclosed fence opened at line {}",
        open_at.unwrap()
    );
    assert!(
        !out.is_empty(),
        "docs/FERMIONICQASM.md has no fenced blocks; the examples are not checked"
    );
    out
}

#[test]
fn every_fenced_block_in_the_language_doc_is_exercised() {
    let md = read_doc();
    let blocks = fenced_blocks(&md);
    let quote_src =
        std::fs::read_to_string(repo_file("tests/fixtures/fermionicqasm/conventions.json"))
            .expect("conventions.json");

    let mut accepted = 0usize;
    let mut refused = 0usize;
    let mut quoted = 0usize;

    for block in &blocks {
        let where_ = format!("docs/FERMIONICQASM.md:{} ({})", block.line, block.info);
        if block.info == "fermionicqasm" {
            accepted += 1;
            let ir = lower_to_ir(&block.body).unwrap_or_else(|e| {
                panic!(
                    "{where_} did not parse and lower:\n{e}\n---\n{}",
                    block.body
                )
            });
            assert_eq!(
                ir.circuit_type,
                CircuitType::Fermionic,
                "{where_} lowered as {:?}, not Fermionic",
                ir.circuit_type
            );
            assert!(
                ir.num_qubits > 0,
                "{where_} allocated no modes\n{}",
                block.body
            );
        } else if let Some(needle) = block.info.strip_prefix("fermionicqasm-refuse") {
            let needle = needle.trim();
            assert!(
                !needle.is_empty(),
                "{where_}: a refuse block names the error text it must contain"
            );
            refused += 1;
            match lower_to_ir(&block.body) {
                Ok(_) => panic!(
                    "{where_} was accepted. It must be refused, with {needle:?} in the error.\n{}",
                    block.body
                ),
                Err(err) => assert!(
                    err.contains(needle),
                    "{where_} was refused, but the error does not contain {needle:?}:\n{err}\n---\n{}",
                    block.body
                ),
            }
        } else if block.info == "conventions-quote" {
            quoted += 1;
            for line in block.body.lines().filter(|l| !l.trim().is_empty()) {
                assert!(
                    quote_src.contains(line),
                    "{where_}: `{line}` is not a line of conventions.json. \
                     The document is quoting a pin it does not match."
                );
            }
        } else {
            panic!(
                "{where_}: fence `{}` is not checked. Tag it fermionicqasm, \
                 fermionicqasm-refuse <error text>, or conventions-quote.",
                block.info
            );
        }
    }

    assert_eq!(
        (accepted, refused, quoted),
        (ACCEPT, REFUSE, QUOTE),
        "the language doc gained or lost an example. Update the counts in this \
         test when that is deliberate; a silent drop is an example nobody runs."
    );
}
