// SPDX-License-Identifier: Apache-2.0
//! Regression tests for failure modes that are SILENT — each of these once
//! produced a wrong answer with no error, which is the only kind of bug that
//! survives a passing test suite.

use aria_tune::*;

#[test]
fn non_finite_scores_never_reach_json_or_best() {
    // A diverged run reports NaN. Previously that became a Complete trial,
    // serialised as the invalid JSON literal `NaN`, and could win best()
    // purely by position (NaN compares false against everything).
    let mut s = Study::new(Space::new().int("n", 1, 3, 1), Direction::Maximize);
    let a = s.ask();
    s.tell(a.id, f64::NAN);
    let b = s.ask();
    s.tell(b.id, f64::INFINITY);
    let c = s.ask();
    s.tell(c.id, 0.25);

    let json = s.to_json();
    assert!(!json.contains("NaN"), "invalid JSON literal NaN: {json}");
    assert!(!json.contains("inf"), "invalid JSON literal inf: {json}");
    // Only the finite trial carries a score field.
    assert_eq!(json.matches("\"score\"").count(), 1, "{json}");

    // The finite trial is the only completion, so it must be the best.
    assert_eq!(s.n_complete(), 1);
    assert_eq!(s.n_pruned(), 2);
    assert_eq!(s.best().map(|t| t.id), Some(c.id));
    assert_eq!(s.mean_score(), Some(0.25));
}

#[test]
fn a_lone_nan_trial_yields_no_best() {
    let mut s = Study::new(Space::new().int("n", 1, 3, 1), Direction::Maximize);
    let a = s.ask();
    s.tell(a.id, f64::NAN);
    assert!(s.best().is_none(), "a NaN trial was reported as the best");
    assert!(s.mean_score().is_none());
}

#[test]
#[should_panic(expected = "LogFloat bounds must be strictly positive")]
fn log_float_rejects_a_non_positive_lower_bound() {
    // Previously this produced Float(NaN) at EVERY grid point, silently.
    let _ = Space::new().log_float("lr", 0.0, 1.0, 3);
}

#[test]
#[should_panic(expected = "Int step must be ≥ 1")]
fn int_rejects_a_non_positive_step() {
    // Previously `step: 0` and `step: -2` both silently became step 1,
    // handing back a different space than was asked for.
    let _ = Space::new().int("n", 0, 10, 0);
}

#[test]
fn try_add_reports_instead_of_panicking() {
    let e = Space::new()
        .try_add(
            "lr",
            Param::LogFloat {
                lo: -1.0,
                hi: 1.0,
                res: 3,
            },
        )
        .unwrap_err();
    assert!(e.contains("strictly positive"), "{e}");
    assert!(Space::new()
        .try_add(
            "n",
            Param::Int {
                lo: 1,
                hi: 4,
                step: 1
            }
        )
        .is_ok());
}

#[test]
fn a_hand_built_bad_log_float_still_never_yields_nan() {
    // `Param` is a public enum, so validation at the Space builder is not the
    // only path in. value() must not emit NaN even when constructed directly.
    let p = Param::LogFloat {
        lo: 0.0,
        hi: 1.0,
        res: 4,
    };
    for i in 0..4 {
        let v = p.value(i).as_float().unwrap();
        assert!(v.is_finite(), "grid point {i} is {v}");
    }
}

#[test]
fn json_is_parseable_for_a_realistic_study() {
    // Cheap structural check without pulling in a JSON dependency: balanced
    // braces/brackets and no bare non-finite literals.
    let space = Space::new()
        .int("n", 2, 8, 2)
        .log_float("lr", 1e-3, 1e-1, 4)
        .categorical("opt", &["gd", "adam"]);
    let mut s = Study::new(space, Direction::Maximize).with_sampler(Box::new(TpeSampler::new(3)));
    for i in 0..12 {
        let t = s.ask();
        s.tell(t.id, if i == 4 { f64::NAN } else { 0.1 * i as f64 });
    }
    let j = s.to_json();
    let braces = j.chars().filter(|c| *c == '{').count() as i64
        - j.chars().filter(|c| *c == '}').count() as i64;
    let brackets = j.chars().filter(|c| *c == '[').count() as i64
        - j.chars().filter(|c| *c == ']').count() as i64;
    assert_eq!(braces, 0, "unbalanced braces: {j}");
    assert_eq!(brackets, 0, "unbalanced brackets: {j}");
    assert!(!j.contains("NaN") && !j.contains(":inf"), "{j}");
}

#[test]
fn nan_bounds_are_rejected_not_silently_accepted() {
    // `NaN <= 0.0` is false, so a naive positivity check would let a NaN
    // bound through and put NaN at every grid point. Finiteness is checked
    // first for exactly that reason.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let e = Space::new()
            .try_add(
                "lr",
                Param::LogFloat {
                    lo: bad,
                    hi: 1.0,
                    res: 3,
                },
            )
            .unwrap_err();
        assert!(e.contains("finite"), "bound {bad} gave: {e}");
        let e2 = Space::new()
            .try_add(
                "x",
                Param::Float {
                    lo: 0.0,
                    hi: bad,
                    res: 3,
                },
            )
            .unwrap_err();
        assert!(e2.contains("finite"), "Float bound {bad} gave: {e2}");
    }
}
