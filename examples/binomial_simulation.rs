use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

struct BernoulliThreshold {
    target: f64,
    rng: StdRng,
}

impl RootOracle<f64> for BernoulliThreshold {
    fn evaluate(&mut self, x: f64) -> f64 {
        let p = x.clamp(0.0, 1.0);

        let success = self.rng.random_bool(p);

        (success as u8 as f64) - self.target
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let oracle = BernoulliThreshold {
        target: 0.4,
        rng: StdRng::seed_from_u64(7),
    };

    let config = BisectorConfig {
        max_observations: 300,
        max_knots: 2_000,
        max_sign_evaluations: 500,
        rel_tol: 1e-3,
        tolerance_window: 10,
    };

    let result = run(0.0..1.0, ConfidenceLevel::new(0.8)?, oracle, config)?;

    println!("true root ≈ 0.4");
    println!("interval: {:?}", result.interval);

    Ok(())
}
