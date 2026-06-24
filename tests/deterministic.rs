use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};

struct LinearRoot {
    root: f64,
}

impl RootOracle<f64> for LinearRoot {
    fn evaluate(&mut self, x: f64) -> f64 {
        x - self.root
    }
}

#[test]
fn finds_root_of_deterministic_linear_function() {
    let root = 2.5;

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
        LinearRoot { root },
        config,
    )
    .unwrap();

    let interval = output.interval;

    assert!(interval.lower() <= root);
    assert!(interval.upper() >= root);
    assert!(interval.width() < 0.001);
}

#[test]
fn works_on_negative_domain() {
    let root = -3.0;

    let config = BisectorConfig {
        max_observations: 10000,
        max_knots: 1000,
        max_sign_evaluations: 1000,
        rel_tol: 1e-5,
        tolerance_window: 10,
    };

    let output = run(
        -10.0..-1.0,
        ConfidenceLevel::new(0.8).unwrap(),
        LinearRoot { root },
        config,
    )
    .unwrap();

    let interval = output.interval;

    assert!(interval.lower() <= root);
    assert!(interval.upper() >= root);
    assert!(interval.width() < 0.001);
}

#[test]
fn works_on_log_scaled_positive_domain() {
    let root = 1e-3;

    let config = BisectorConfig {
        max_observations: 10000,
        max_knots: 1000,
        max_sign_evaluations: 1000,
        rel_tol: 1e-5,
        tolerance_window: 10,
    };

    let output = run(
        1e-9..1e3,
        ConfidenceLevel::new(0.8).unwrap(),
        LinearRoot { root },
        config,
    )
    .unwrap();

    let interval = output.interval;

    assert!(interval.lower() <= root);
    assert!(interval.upper() >= root);
}

struct NoisyLinear {
    root: f64,
    noise: Vec<f64>,
    index: usize,
}

impl RootOracle<f64> for NoisyLinear {
    fn evaluate(&mut self, x: f64) -> f64 {
        let eps = self.noise[self.index % self.noise.len()];
        self.index += 1;
        x - self.root + eps
    }
}

struct NoRoot;

impl RootOracle<f64> for NoRoot {
    fn evaluate(&mut self, _x: f64) -> f64 {
        1.0
    }
}

#[test]
fn reports_no_root_when_domain_does_not_bracket_root() {
    let config = BisectorConfig {
        max_observations: 10000,
        max_knots: 1000,
        max_sign_evaluations: 1000,
        rel_tol: 1e-5,
        tolerance_window: 10,
    };

    let result = run(0.0..1.0, ConfidenceLevel::new(0.8).unwrap(), NoRoot, config);

    assert!(result.is_err());

    let err = result.unwrap_err();
}
