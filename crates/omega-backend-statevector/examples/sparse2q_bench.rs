//! A/B timing for the sparse 2q fast paths.
//!
//! Two defences against measuring the harness instead of the kernel, both
//! learned from a first draft of this file that reported a confident 2.5x
//! REGRESSION for a change that is a speedup:
//!
//! 1. **Warm up, outside the timed region.** The first pass over a 2^n
//!    `Complex64` vector pays first-touch page faults for the whole
//!    allocation — 64 MB at n=22. The draft attributed all of that to
//!    whichever gate happened to run first.
//! 2. **Interleave the gate order across repetitions.** The contamination
//!    above is POSITIONAL, not per-gate, so anything that only ever runs the
//!    gates in a fixed order can absorb it into whichever gate leads and
//!    produce a monotone curve that reads exactly like a real ordering effect.
//!    Rotating the order spreads any positional cost evenly, so it shows up as
//!    noise in every arm rather than as signal in one. Cheap insurance.
//!
//! The tell that something is wrong: with the fast paths OFF, every gate here
//! runs the same dense kernel and the baselines should be indistinguishable.
//! A baseline that varies across gates is measuring something other than the
//! gate.
use num_complex::Complex64;
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(22);
    let reps: usize = 30;
    let rounds: usize = 5;
    let dim = 1usize << n;
    let mut state: Vec<Complex64> = (0..dim)
        .map(|i| Complex64::new((i % 97) as f64 * 1e-6, (i % 89) as f64 * 1e-6))
        .collect();

    let gates: Vec<(&str, [Complex64; 16])> = vec![
        ("cz  ", omega_backend_statevector::gates::cz()),
        ("crz ", omega_backend_statevector::gates::crz(0.37)),
        ("swap", omega_backend_statevector::gates::swap()),
        ("cy  ", omega_backend_statevector::gates::cy()),
        ("cu3 ", omega_backend_statevector::gates::cu3(0.3, 0.5, 0.7)),
        // TWO CONTROLS that can never take a fast path: `rbs` is a rotation in
        // the {01,10} subspace — not diagonal, not a permutation, not
        // controlled — and `dense` is a full matrix with no zero and no
        // repeat. They run the same dense kernel as each other in EVERY
        // configuration, so their timings must agree. Asserted at the end.
        ("rbs ", omega_backend_statevector::gates::rbs(0.41)),
        // Both controls must be gates NO dispatch can claim. `rbs` was one
        // until it got the middle-block path — a control that quietly acquires
        // a fast path stops being a control, so these are two distinct FULL
        // matrices with no zero and no repeat, which nothing can pattern-match.
        ("dnsA*", DENSE_CONTROL),
        ("dnsB*", DENSE_CONTROL_B),
    ];

    // Touch every page once, and warm each gate, all outside timing.
    for (r, s) in state.iter_mut().enumerate() {
        s.re += (r & 1) as f64 * 1e-18;
    }
    for (_, g) in &gates {
        for r in 0..5 {
            omega_backend_statevector::sim::apply_2q_pub(&mut state, n, (r % 3) + 1, 0, g);
        }
    }

    let mut best = vec![f64::INFINITY; gates.len()];
    for round in 0..rounds {
        // Rotate the starting gate each round: positional cost cannot settle
        // on one arm.
        for off in 0..gates.len() {
            let gi = (off + round) % gates.len();
            let (_, g) = &gates[gi];
            let t = Instant::now();
            for r in 0..reps {
                omega_backend_statevector::sim::apply_2q_pub(&mut state, n, (r % 3) + 1, 0, g);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            best[gi] = best[gi].min(ms);
        }
    }
    for (i, (label, _)) in gates.iter().enumerate() {
        println!(
            "  {label}  n={n}  best-of-{rounds}  {:>7.3} ms/gate",
            best[i]
        );
    }

    // THE HARNESS CHECKS ITSELF. The two starred arms run the same dense
    // kernel, so any gap between them is the bench measuring its own
    // scheduling, cache state or page faults rather than the gate — and once
    // that is happening, no speedup on the other arms is trustworthy either.
    //
    // An assertion, not a printed number, on purpose: the earlier draft of
    // this file DID show its contamination — as a monotone curve across the
    // arms — and it was eyeballed and believed. A number you have to notice
    // is a number that gets noticed once.
    let (a, b) = (best[gates.len() - 2], best[gates.len() - 1]);
    let spread = (a - b).abs() / a.min(b);
    println!(
        "\n  control arms (both always dense): {a:.3} vs {b:.3} ms — spread {:.1}%",
        spread * 100.0
    );
    assert!(
        spread < 0.15,
        "BENCH IS UNSOUND: two arms running the identical dense kernel differ by \
         {:.1}% ({a:.3} vs {b:.3} ms). Something other than the gate is being \
         measured — do not trust any speedup from this run.",
        spread * 100.0
    );
    println!("  bench self-check OK");
}

/// A fixed full 4x4 with no zero and no repeat, so it can never be mistaken
/// for diagonal, a permutation, or controlled.
const DENSE_CONTROL_B: [Complex64; 16] = {
    let mut m = [Complex64::new(0.0, 0.0); 16];
    let mut i = 0;
    while i < 16 {
        m[i] = Complex64::new(0.23 - i as f64 * 0.041, 0.17 + i as f64 * 0.029);
        i += 1;
    }
    m
};

const DENSE_CONTROL: [Complex64; 16] = {
    let mut m = [Complex64::new(0.0, 0.0); 16];
    let mut i = 0;
    while i < 16 {
        m[i] = Complex64::new(0.11 + i as f64 * 0.037, -0.07 - i as f64 * 0.019);
        i += 1;
    }
    m
};
