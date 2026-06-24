# probabilistic_bisector

## Probabilistic Bisector

`probabilistic_bisector` provides a probabilistic bisection algorithm for
locating roots of scalar objective functions observed in noise.

Classical bisection assumes that evaluating an objective function gives an
exact sign. In many numerical, simulation, and experimental settings this is
not true: repeated evaluations at the same point may produce different
values, and the sign of the objective may only be inferable statistically.

This crate is designed for that setting.

### Motivation

Suppose we want to find a root `x*` of a scalar objective function

`text
f(x*) = 0
`

but each call to the objective produces a noisy observation. A single
evaluation may not reliably tell us whether `f(x)` is positive or negative.

Instead of treating each sign observation as exact, this crate maintains a
posterior distribution over the possible root location. Each observation
updates that distribution, and the algorithm returns a confidence interval
for the root.

### Theoretical basis

The implementation follows the probabilistic bisection framework described
by Waeber.

The algorithm maintains a posterior distribution over the root location on a
fixed internal domain. At each step:

1. A query point is selected from the current posterior.
2. The objective is evaluated repeatedly at that point.
3. A curved-boundary sign test determines the sign of the objective to the
   requested confidence level.
4. The posterior mass is updated according to whether the root is expected
   to lie to the left or right of the query point.
5. A sequential confidence interval is updated by intersecting the previous
   interval with the current Waeber-style confidence region.

Internally, posterior mass is stored over a partition of the search domain.
The posterior is represented in log-space for numerical stability.

### Coordinate scaling

The posterior is represented on a numerically convenient internal domain,
usually `[0, 1]`. User objective functions are evaluated on their original
raw domain.

A `Scaler` maps between these coordinate systems. For strictly positive
domains spanning several orders of magnitude, logarithmic scaling may be used
so that posterior resolution is distributed multiplicatively rather than
linearly.

### Implementing an objective

Problems are represented by implementing [`RootOracle`].

The oracle provides noisy evaluations of the objective. Implementations may
be deterministic or stochastic.

```rust
use confi::ConfidenceLevel;
use probabilistic_bisector::{run, BisectorConfig, RootOracle};

struct Linear {
    root: f64,
}

impl RootOracle<f64> for Linear {
    fn evaluate(&mut self, x: f64) -> f64 {
        x - self.root
    }
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BisectorConfig {
        max_observations: 10000,      // Maximum observations in the loop
        max_knots: 1000,              // Maximum knots in the posterior distribution
        max_sign_evaluations: 1000,   // Maximum function calls in evaluating the objective sign
        rel_tol: 1e-5,                // Target relative tolerance
        tolerance_window: 10,         // Number of evaluations relative tolerance should be stable
    };

    let result = run(
        0.0..10.0,                    // Search range
        ConfidenceLevel::new(0.8)?,   // Required confidence level
        Linear { root: 2.5 },         // Problem
        config,
    )?;

    println!("root interval: {:?}", result.interval);
    println!("termination: {:?}", result.termination);
    Ok(())
}
```

### Stochastic objectives

A `RootOracle` may use internal randomness. Since [`RootOracle::evaluate`]
takes `&mut self`, objectives can store their own random number generator,
allowing reproducible seeded tests.

`rust
use probabilistic_bisector::RootOracle;

struct NoisyLinear {
    root: f64,
    index: usize,
}

impl RootOracle<f64> for NoisyLinear {
    fn evaluate(&mut self, x: f64) -> f64 {
        let noise = if self.index % 2 == 0 { 0.001 } else { -0.001 };
        self.index += 1;
        x - self.root + noise
    }
}
`

### Termination

The solver may terminate because:

- the requested tolerance was reached,
- the maximum iteration budget was reached,
- or the objective sign became indeterminate at all useful query points.

Sign indeterminacy is not necessarily an error. It usually means that the
algorithm has reached the noise floor of the objective: additional samples at
nearby points do not determine the sign reliably within the configured
evaluation budget -> Given the noise in the objective function the requested
tolerance might be unachievable.

### Output

Successful runs return a result containing:

- a confidence interval for the root,
- execution summary information,
- and the reason the solver terminated.

Exceptional failures are reserved for invalid inputs, invalid oracle values,
or internal invariant violations.

License: MIT
