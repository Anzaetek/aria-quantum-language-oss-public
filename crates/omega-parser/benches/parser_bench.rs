//! Parser bench: lower a 1000-line QASM file to CircuitIR.

use criterion::{criterion_group, criterion_main, Criterion};

use omega_parser::lower_to_ir;

fn build_large_qasm(lines: usize) -> String {
    let mut s = String::with_capacity(lines * 32);
    s.push_str("OPENQASM 2.0;\n");
    s.push_str("include \"qelib1.inc\";\n");
    s.push_str("qreg q[20];\n");
    s.push_str("creg c[20];\n");

    // Fill with alternating rotation + CX pairs.
    for i in 0..lines {
        let q = i % 20;
        match i % 4 {
            0 => s.push_str(&format!("rx(0.1) q[{}];\n", q)),
            1 => s.push_str(&format!("ry(0.2) q[{}];\n", q)),
            2 => s.push_str(&format!("rz(0.3) q[{}];\n", q)),
            _ => s.push_str(&format!("cx q[{}], q[{}];\n", q, (q + 1) % 20)),
        }
    }
    s
}

fn bench_parser(c: &mut Criterion) {
    let source = build_large_qasm(1000);

    c.bench_function("parser_1000line_qasm", |b| {
        b.iter(|| {
            let ir = lower_to_ir(&source).unwrap();
            std::hint::black_box(ir);
        });
    });
}

criterion_group!(benches, bench_parser);
criterion_main!(benches);
