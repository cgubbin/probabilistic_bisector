use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};

struct AlternatingNoise {
    root: f64,
    amplitude: f64,
    i: usize,
}

impl RootOracle<f64> for AlternatingNoise {
    fn evaluate(&mut self, x: f64) -> f64 {
        let eps = if self.i.is_multiple_of(2) {
            self.amplitude
        } else {
            -self.amplitude
        };
        self.i += 1;

        x - self.root + eps
    }
}

#[test]
fn converges_with_small_deterministic_noise() {
    let root = 0.37;

    let config = BisectorConfig {
        max_observations: 10000,
        max_knots: 1000,
        max_sign_evaluations: 1000,
        rel_tol: 1e-5,
        tolerance_window: 10,
    };

    let output = run(
        0.0..100.0,
        ConfidenceLevel::new(0.8).unwrap(),
        AlternatingNoise {
            root,
            amplitude: 0.0001,
            i: 0,
        },
        config,
    )
    .unwrap();

    assert!(output.interval.lower() <= root);
    assert!(output.interval.upper() >= root);
    assert!(output.interval.width() < 0.001);
}
