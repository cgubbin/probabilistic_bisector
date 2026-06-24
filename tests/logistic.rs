use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};

struct LogisticMean {
    root: f64,
    scale: f64,
    i: usize,
}

impl RootOracle<f64> for LogisticMean {
    fn evaluate(&mut self, x: f64) -> f64 {
        // Smooth monotone signal, deterministic pseudo-noise.
        let signal = ((x - self.root) / self.scale).tanh();
        let noise = (((self.i * 48271) % 1000) as f64 / 1000.0) - 0.5;
        self.i += 1;

        signal + 0.15 * noise
    }
}

#[test]
fn handles_saturating_monotone_objective() {
    let root = 2.5;
    let config = BisectorConfig {
        max_observations: 1000,
        max_knots: 1000,
        max_sign_evaluations: 1000,
        rel_tol: 1e-5,
        tolerance_window: 10,
    };

    let out = run(
        0.0..10.0,
        ConfidenceLevel::new(0.8).unwrap(),
        LogisticMean {
            root,
            scale: 0.8,
            i: 0,
        },
        config,
    )
    .unwrap();

    let interval = out.interval;

    assert!(interval.lower() <= root);
    assert!(interval.upper() >= root);
    assert!(interval.width() < 0.1);
}
