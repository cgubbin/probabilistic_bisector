use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};

struct HeteroscedasticNoise {
    root: f64,
    i: usize,
}

impl RootOracle<f64> for HeteroscedasticNoise {
    fn evaluate(&mut self, x: f64) -> f64 {
        let phase = ((self.i * 17) % 11) as f64 - 5.0;
        self.i += 1;

        let noise_scale = 0.02 + 0.15 * (x - self.root).abs();
        x - self.root + phase * noise_scale / 10.0
    }
}

#[test]
fn handles_noise_that_depends_on_position() {
    let root = 0.63;

    let config = BisectorConfig {
        max_observations: 10000,
        max_knots: 1000,
        max_sign_evaluations: 1000,
        rel_tol: 1e-5,
        tolerance_window: 10,
    };

    let output = run(
        0.0..10.0,
        ConfidenceLevel::new(0.8).unwrap(),
        HeteroscedasticNoise { root, i: 0 },
        config,
    )
    .unwrap();

    let interval = output.interval;

    assert!(interval.lower() <= root);
    assert!(interval.upper() >= root);
    assert!(interval.width() < 0.01);
}
