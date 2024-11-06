use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor},
    solver::brent::BrentRoot,
};
use argmin_observer_slog::SlogLogger;
use confi::{Confidence, ConfidenceLevel};
use probabalistic_bisector::{Bisectable, ProbabalisticBisector};
use rand::distributions::Distribution;
use statrs::distribution::Normal;
use trellis_runner::GenerateBuilder;

/// Test function generalise from Wikipedia example
#[derive(Clone, Debug)]
struct TestFunc {
    zero1: f64,
    zero2: f64,
}

impl CostFunction for TestFunc {
    // one dimensional problem, no vector needed
    type Param = f64;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok((p + self.zero1) * (p - self.zero2) * (p - self.zero2))
    }
}

struct NoisyTestFunc<F> {
    deterministic: F,
    noise_kernel: Normal,
}

impl<F: CostFunction<Param = f64, Output = f64>> Bisectable<f64> for NoisyTestFunc<F> {
    fn evaluate(&self, x: f64) -> f64 {
        self.deterministic.cost(&x).unwrap() + self.noise_kernel.sample(&mut rand::thread_rng())
    }
}

#[test]
fn brent_comparison() {
    let deterministic = TestFunc {
        zero1: 3.,
        zero2: -1.,
    };
    let init_param = 0.5;
    let solver = BrentRoot::new(-4., 1., 1e-11);

    let brent = Executor::new(deterministic.clone(), solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    let stochastic = NoisyTestFunc {
        deterministic,
        noise_kernel: Normal::new(0.0, 0.001).unwrap(),
    };
    let domain = -4.0..1.0;
    let bisector = ProbabalisticBisector::new(domain, ConfidenceLevel::ninety_nine_percent());

    let runner = bisector
        .build_for(stochastic)
        .configure(|state| state.max_iters(1000).relative_tolerance(1e-3))
        .finalise()
        .unwrap();

    let probabalistic = runner.run().unwrap().result;

    assert!(probabalistic.interval.contains(brent.state.param.unwrap()));
}
