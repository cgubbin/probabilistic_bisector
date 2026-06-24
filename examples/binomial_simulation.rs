use confi::ConfidenceLevel;
use probabilistic_bisector::{BisectorConfig, RootOracle, run};
use rand::{RngExt, SeedableRng, rngs::StdRng};

struct BernoulliThreshold {
    target: f64,
    trials_per_eval: usize,
    rng: StdRng,
}

impl RootOracle<f64> for BernoulliThreshold {
    fn evaluate(&mut self, x: f64) -> f64 {
        let p = x.clamp(0.0, 1.0);

        let successes = (0..self.trials_per_eval)
            .filter(|_| self.rng.random_bool(p))
            .count();

        let estimate = successes as f64 / self.trials_per_eval as f64;

        estimate - self.target
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let oracle = BernoulliThreshold {
        target: 0.4,
        trials_per_eval: 50,
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
