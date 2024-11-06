# probabilistic bisection
---

An implementation of linear bisection for finding the roots of a stochastic function. The current implementation only works for functions with one root in the search range of interest.

Root-finding is a common problem in scientific computation. When the function to be studied is deterministic a simple linear bisection algorithm or something more sophisticated such as Brent's method can be applied. These algorithms rely on the deterministic nature of the objective function: if they evaluate the function multiple times for a given input they expect the output to be the same in each case.

Stochastic functions do not behave in this way. Their output is noisy, and can vary on repeated evaluation. Often though we still want to find a point $x$ such that the function $g\left(x\right) = 0$. In the presence of noise this can only be achieved by identifying a confidence interval surrounding the root which is progressively narrowed.

This crate identifies the roots of stochastic 1D functions by implementation of a probabilistic bisection algorithm as described in the PhD thesis of [R. Waeber](https://people.orie.cornell.edu/shane/theses/ThesisRolfWaeber.pdf). The algorithm works by querying the objective function $g$ to determine whether the root lies to the left or right of a test point $t$. Each query has probability $1 - p(t)$ of being incorrect due to the stochastic nature of $g$. The algorithm accounts for this by creating a probability distribution containing knowledge of the true value of the root.

## Usage
---

A target objective function must implement the `Bisectable` trait. This has one user-facing method, `evaluate`. It is expected that for implementors of `Bisectable` the `evaluate` method is noisy. 
```rust
use probabilistic_bisector::{Bisectable, ProbabilisticBisector, GenerateBuilder, Confidence, ConfidenceLevel};

struct Linear {
    gradient: f64,
    intercept: f64,
}

impl Bisectable<f64> for Linear {
    // This function can be noisy!
    fn evaluate(&self, x: f64) -> f64 {
        self.gradient * x + self.intercept 
    }
}

let problem = Linear { gradient: 1.0, intercept: 0.0 };

let domain = -1.0..1.0;
let bisector = ProbabilisticBisector::new(domain, ConfidenceLevel::ninety_nine_percent());

let runner = bisector
    .build_for(problem)
    .configure(|state| state.max_iters(1000).relative_tolerance(1e-3))
    .finalise()
    .unwrap();

let result = runner.run().unwrap().result;

assert!(result.interval.contains(0.0));

```
