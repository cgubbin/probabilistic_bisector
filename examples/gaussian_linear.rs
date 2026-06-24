use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

struct GaussianLinear {
    root: f64,
    rng: StdRng,
    noise: Normal<f64>,
}

impl RootOracle<f64> for GaussianLinear {
    fn evaluate(&mut self, x: f64) -> f64 {
        x - self.root + self.noise.sample(&mut self.rng)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = 0.37;

    let oracle = GaussianLinear {
        root,
        rng: StdRng::seed_from_u64(42),
        noise: Normal::new(0.0, 0.01)?,
    };

    let config = BisectorConfig {
        max_observations: 300,
        max_knots: 2_000,
        max_sign_evaluations: 500,
        rel_tol: 1e-3,
        tolerance_window: 10,
    };

    let result = run(0.0..1.0, ConfidenceLevel::new(0.8)?, oracle, config)?;

    println!("true root: {root}");
    println!("interval: {:?}", result.interval);
    println!("termination: {:?}", result.termination);

    Ok(())
}
