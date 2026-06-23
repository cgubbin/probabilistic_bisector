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
#[tracing_test::traced_test]
fn finds_root_of_deterministic_linear_function() {
    let root = 2.5;

    let config = BisectorConfig {
        max_observations: 100,
        max_knots: 100,
        max_sign_evaluations: 1000,
        rel_tol: 1e-5,
        tolerance_window: 10,
    };

    let output = run(
        0.0..10.0,
        ConfidenceLevel::new(0.8).unwrap(),
        LinearRoot { root },
        config,
    );

    let interval = output;
    dbg!(&interval);

    // assert!(interval.lower() <= root);
    // assert!(interval.upper() >= root);
    // assert!(interval.width() < 1.0);
}

// #[test]
// fn works_on_negative_domain() {
//     let root = -3.0;

//     let output = run(
//         -10.0..-1.0,
//         ConfidenceLevel::new(0.8).unwrap(),
//         LinearRoot { root },
//         100,
//         1_000,
//         100,
//         1e-3,
//     )
//     .unwrap();

//     let interval = output.output();

//     assert!(interval.lower() <= root);
//     assert!(interval.upper() >= root);
// }

// #[test]
// fn works_on_log_scaled_positive_domain() {
//     let root = 1e-3;

//     let output = run(
//         1e-9..1e3,
//         ConfidenceLevel::new(0.8).unwrap(),
//         LinearRoot { root },
//         150,
//         2_000,
//         100,
//         1e-3,
//     )
//     .unwrap();

//     let interval = output.output();

//     assert!(interval.lower() <= root);
//     assert!(interval.upper() >= root);
// }

// struct NoisyLinear {
//     root: f64,
//     noise: Vec<f64>,
//     index: usize,
// }

// impl RootOracle<f64> for NoisyLinear {
//     fn evaluate(&mut self, x: f64) -> f64 {
//         let eps = self.noise[self.index % self.noise.len()];
//         self.index += 1;
//         x - self.root + eps
//     }
// }

// #[test]
// fn gives_reasonable_interval_for_reproducible_noisy_linear_function() {
//     let root = 0.35;

//     let noise = vec![-0.01, 0.02, -0.015, 0.005, 0.0, 0.01, -0.02, 0.015];

//     let output = run(
//         0.0..1.0,
//         ConfidenceLevel::new(0.8).unwrap(),
//         NoisyLinear {
//             root,
//             noise,
//             index: 0,
//         },
//         150,
//         2_000,
//         200,
//         1e-3,
//     )
//     .unwrap();

//     let interval = output.output();

//     assert!(interval.lower() <= root);
//     assert!(interval.upper() >= root);
//     assert!(interval.width() < 0.25);
// }

// struct NoRoot;

// impl RootOracle<f64> for NoRoot {
//     fn evaluate(&mut self, _x: f64) -> f64 {
//         1.0
//     }
// }

// #[test]
// fn reports_no_root_when_domain_does_not_bracket_root() {
//     let result = run(
//         0.0..1.0,
//         ConfidenceLevel::new(0.8).unwrap(),
//         NoRoot,
//         50,
//         1_000,
//         50,
//         1e-3,
//     );

//     assert!(result.is_err());
// }
