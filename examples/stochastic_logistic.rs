use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

struct LogisticResponse {
    root: f64,
    scale: f64,
    rng: StdRng,
    noise: Normal<f64>,
}

impl RootOracle<f64> for LogisticResponse {
    fn evaluate(&mut self, x: f64) -> f64 {
        let signal = ((x - self.root) / self.scale).tanh();
        signal + self.noise.sample(&mut self.rng)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = 2.5;

    let oracle = LogisticResponse {
        root,
        scale: 0.5,
        rng: StdRng::seed_from_u64(123),
        noise: Normal::new(0.0, 0.05)?,
    };

    let config = BisectorConfig {
        max_observations: 300,
        max_knots: 2_000,
        max_sign_evaluations: 500,
        rel_tol: 1e-3,
        tolerance_window: 10,
    };

    let result = run(0.0..10.0, ConfidenceLevel::new(0.8)?, oracle, config)?;

    println!("true root: {root}");
    println!("interval: {:?}", result.interval);

    Ok(())
}
