//! This crate implements linear bisection for finding the root of a one-dimensional stochastic function.
//!
//! Root finding is a fairly simple problem when the function we are looking for the root of is
//! deterministic. We can just use a linear bisection algorithm, or something more sophisticated
//! like Brent's method.
//!
//! In a general stochastic root finding method our goal is still the same: we want to locate a
//! point x such that the optimisation function g(x) = 0 when the optimisation function can only be
//! observed in the presence of noise.
//!
//! This module achieves this by implementing the probabilistic bisection algorithm. This queries
//! the function g as to whether the root lies to the left or right of a prescribed point t. As a
//! consequence of observational noise each query has probability 1 - p(t) of being incorrect. To
//! account for this the algorithm updates a probability distribution which attempts to represent
//! knowledge of the true root x. A full description is available in the PhD thesis of [R. Waeber](https://people.orie.cornell.edu/shane/theses/ThesisRolfWaeber.pdf).
//!
//! ```rust
//! use probabilistic_bisector::{Bisectable, ProbabilisticBisector, GenerateBuilder, Confidence,
//! ConfidenceLevel};
//!
//! struct Linear {
//!     gradient: f64,
//!     intercept: f64,
//! }
//!
//! impl Bisectable<f64> for Linear {
//!     // This function can be noisy!
//!     fn evaluate(&self, x: f64) -> f64 {
//!         self.gradient * x + self.intercept
//!     }
//! }
//!
//! let problem = Linear { gradient: 1.0, intercept: 0.0 };
//!
//! let domain = -1.0..1.0;
//! let bisector = ProbabilisticBisector::new(domain, ConfidenceLevel::ninety_nine_percent());
//!
//! let runner = bisector
//!     .build_for(problem)
//!     .configure(|state| state.max_iters(1000).relative_tolerance(1e-3))
//!     .finalise()
//!     .unwrap();
//!
//! let result = runner.run().unwrap().result;
//!
//! assert!(result.interval.contains(0.0));
//!
//! ```

// In the PBA we cannot be certain of the true location of the root. Instead we track a
// probability distribution which represents the sum of our knowledge about it's best location.
//
// The `distribution` module contains an implementation of this distribution. The class is initialised
// with a range of values which can be expected to contain the root with a probability approaching
// unity. It stores the range limits, or samples, and the logarithm of the probability in the
// intermediate region ln(1) = 0. As the algorithm proceeds, new sample points are inserted and
// the probabilities are updated.
mod distribution;

// The result of a probabilistic bisection is a confidence interval. This expresses a statistical
// certainty about the range of values enclosing the true value of the root of the objective
// function.
//
// This module contains methods to compute and update confidence intervals as described in §3.3
mod interval;

mod error;

use confi::SignificanceLevel;
use distribution::{Distribution, DistributionError};
pub use error::Error;
pub(crate) use interval::CombinedConfidenceInterval;
use interval::ConfidenceIntervals;
use num_traits::{Float, FromPrimitive};
use std::{fmt, iter, ops::Range};
use trellis_runner::{Calculation, Problem, TrellisFloat, UserState};

pub use confi::{Confidence, ConfidenceLevel};
pub use trellis_runner::GenerateBuilder;

// Representing the sign of a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Positive,
    Negative,
    Zero,
}

impl<T: Float> From<T> for Sign {
    fn from(x: T) -> Self {
        if x.is_sign_positive() {
            Sign::Positive
        } else if x.is_sign_negative() {
            Sign::Negative
        } else {
            Sign::Zero
        }
    }
}

impl Sign {
    fn signum<T: Float>(&self) -> T {
        match self {
            Sign::Positive => T::one(),
            Sign::Negative => T::one().neg(),
            Sign::Zero => T::one(),
        }
    }
}

// Trait for bisectable objective functions
pub trait Bisectable<T: Float + FromPrimitive + std::fmt::Debug> {
    // Evaluate the objective function at the given point.
    //
    // This method is expected to be stochastic, meaning it may return different values for the
    // same input value. It is expected to be called through the sign function, which attempts to
    // determine the sign of the objective function at the given point to a specified confidence
    // level.
    fn evaluate(&self, x: T) -> T;

    // The sign of the slope of the function over the domain
    //
    // It is a necessary condition that the function has a root within the domain. This means it is
    // required that the sign of the function changes between the start and end of the domain. We
    // can therefore determine the slope of the function by evaluating the sign at the start and end
    fn slope_sign(
        &self,
        domain: &Range<T>,
        confidence_level: ConfidenceLevel<T>,
        max_iter: usize,
    ) -> Result<Sign, Error<T>> {
        let sign_start = self.sign(domain.start, confidence_level, max_iter)?;
        let sign_end = self.sign(domain.end, confidence_level, max_iter)?;

        match (sign_start, sign_end) {
            (Sign::Positive, Sign::Negative) => Ok(Sign::Negative),
            (Sign::Negative, Sign::Positive) => Ok(Sign::Positive),
            _ => Err(Error::NoRootDetected),
        }
    }

    // The sign of the function at the evaluation point x
    //
    // Recall that the point of this module is that we can only observe the objective function with
    // noise. This means we cannot be certain of the sign of the function at any point. This
    // function computes the sign of the function at the given point x to the prescribed level of
    // confidence.
    //
    // This algorithm will fail if the sign cannot be determined in the prescribed number of
    // iterations.
    //
    // Appendix B of Waeber's PhD thesis describes 3 approaches to determining the sign of the
    // objective function within the prescribed confidence level. We take the approach recommended
    // in the third approach.
    //
    // Take the set of evaluations (\zeta(\theta)) as a set of random variables with mean \theta.
    // We perform a power test, which aims to test whether \theta < \theta_0 or \theta > \theta_0.
    // We do this by observing the random walk S = \sum \zeta(\theta) - \theta_0 until a test is
    // passed.
    //
    // This test is referred to as a curved boundary test. Many test functions are available, but
    // we choose to assume a random walk (Eq B.6) to define the stopping rule.
    fn sign(
        &self,
        x: T,
        confidence_level: ConfidenceLevel<T>,
        max_iter: usize,
    ) -> Result<Sign, Error<T>> {
        let mut random_walk = T::zero();
        let p_c = confidence_level.probability();
        let two = T::one() + T::one();
        let one = T::one();

        for ii in 0..max_iter {
            random_walk = random_walk + self.evaluate(x).signum();
            let n = T::from_usize(ii).unwrap();

            let power_test = ((two * n) * ((n + one).ln() - two.ln() - p_c.ln())).sqrt();

            if random_walk.abs() > power_test {
                return Ok(random_walk.into());
            }
        }

        Err(Error::SignDeterminationError(x.to_f64().unwrap(), max_iter))
    }
}

pub struct ProbabilisticBisector<T> {
    // The range of values expected with high certainty to contain the true value of the root
    domain: Range<T>,
    // The scaler for the domain.
    //
    // The solver operates internally on range [0, 1] to improve convergence. The [`Scaler`]
    // converts values from this range to the true range of function input.
    scaler: Scaler<T>,
    // The maximum number of elements to use in computing the underlying probability distribution
    n_max: usize,
    // The maximum number of iterations
    max_iter: usize,
    // The confidence level
    confidence_level: ConfidenceLevel<T>,
    significance: SignificanceLevel<T>,
}

struct Scaler<T> {
    shift: T,
    factor: T,
    flip_sign: bool,
    log_transform: bool,
}

impl<T: Float + FromPrimitive> Scaler<T> {
    fn new(mut raw: Range<T>) -> Self {
        let mut log_transform = false;
        let mut flip_sign = false;
        let (factor, shift) = match (raw.start.is_sign_positive(), raw.end.is_sign_positive()) {
            // If the range is all positive consider a logarithmic transform
            (true, true) => {
                // If there is a significant disparity between the start and endpoints a
                // logarithmic transform can be useful
                if raw.end / raw.start > T::from_f64(1e3).unwrap() {
                    raw = raw.start.log10()..raw.end.log10();
                    log_transform = true;
                }
                let scaling_factor = (raw.end - raw.start) / T::from_f64(2.0).unwrap();
                let shift = -T::one() - raw.start / scaling_factor;
                (scaling_factor, shift)
            }
            // If the range changes sign just rescale
            (false, true) => {
                let scaling_factor = (raw.end - raw.start) / T::from_f64(2.0).unwrap();
                let shift = -T::one() - raw.start / scaling_factor;
                (scaling_factor, shift)
            }
            (false, false) => {
                flip_sign = true;
                raw = -raw.end..-raw.start;

                if raw.end / raw.start > T::from_f64(1e3).unwrap() {
                    raw = raw.start.log10()..raw.end.log10();
                    log_transform = true;
                }

                let scaling_factor = (raw.end - raw.start) / T::from_f64(2.0).unwrap();
                let shift = -T::one() - raw.start / scaling_factor;
                (scaling_factor, shift)
            }
            _ => unreachable!(
                "if the start is positive and the end is negative the search range is empty."
            ),
        };

        Self {
            shift,
            factor,
            flip_sign,
            log_transform,
        }
    }

    fn shift_sample(&self, sample: T) -> T {
        sample - self.shift
    }

    fn unscale_sample(&self, sample: T) -> T {
        let sign = if self.flip_sign { -T::one() } else { T::one() };
        if self.log_transform {
            return T::from_f64(10.0)
                .unwrap()
                .powf(self.shift_sample(sample) * self.factor * sign);
        }
        self.shift_sample(sample) * self.factor * sign
    }

    fn unscaled_domain(&self) -> Range<T> {
        self.unscale_sample(-T::one())..self.unscale_sample(T::one())
    }
}

impl<T: Float + FromPrimitive + std::fmt::Debug> ProbabilisticBisector<T> {
    pub fn new(domain: Range<T>, confidence_level: ConfidenceLevel<T>) -> Self {
        // TODO: Original impl set the significance to double this value. Does this make sense?
        let significance: SignificanceLevel<T> = confidence_level.into();

        let scaler = Scaler::new(domain.clone());
        let domain = -T::one()..T::one();

        Self {
            domain,
            scaler,
            n_max: 1000,
            max_iter: 100,
            confidence_level,
            significance,
        }
    }
}

pub struct BisectorState<T> {
    confidence: Option<ConfidenceIntervals<T>>,
    distribution: Option<Distribution<T>>,
    slope: Option<Sign>,
    best_width: T,
    prev_best_width: T,
    current_width: T,
}

impl<T> BisectorState<T> {
    fn initialize(
        &mut self,
        domain: Range<T>,
        n_max: usize,
        max_iter: usize,
        slope: Sign,
        confidence_level: ConfidenceLevel<T>,
        significance_level: SignificanceLevel<T>,
    ) -> Result<(), Error<T>>
    where
        T: Float + std::fmt::Debug,
    {
        self.confidence = Some(ConfidenceIntervals::new(
            max_iter,
            confidence_level,
            significance_level,
        ));

        self.distribution = Some(Distribution::new(domain, n_max)?);
        self.slope = Some(slope);
        Ok(())
    }
}

impl<T> UserState for BisectorState<T>
where
    T: Float + FromPrimitive + TrellisFloat + fmt::Debug,
{
    type Float = T;
    type Param = ConfidenceIntervals<T>;

    fn new() -> Self {
        Self {
            confidence: None,
            distribution: None,
            slope: None,
            best_width: T::max_value(),
            prev_best_width: T::max_value(),
            current_width: T::max_value(),
        }
    }

    fn update(&mut self) -> trellis_runner::ErrorEstimate<Self::Float> {
        trellis_runner::ErrorEstimate(self.current_width)
    }

    fn is_initialised(&self) -> bool {
        self.confidence.is_some() && self.slope.is_some() && self.distribution.is_some()
    }

    fn get_param(&self) -> Option<&Self::Param> {
        self.confidence.as_ref()
    }

    fn last_was_best(&mut self) {
        std::mem::swap(&mut self.prev_best_width, &mut self.best_width);
        self.best_width = self.current_width;
    }
}

impl<O, T> Bisectable<T> for Problem<O>
where
    T: Float + FromPrimitive + std::fmt::Debug,
    O: Bisectable<T>,
{
    fn evaluate(&self, x: T) -> T {
        self.as_ref().evaluate(x)
    }
}

impl<O, T> Calculation<O, BisectorState<T>> for ProbabilisticBisector<T>
where
    T: Float + FromPrimitive + fmt::Debug + iter::Sum + TrellisFloat + 'static,
    O: Bisectable<T>,
{
    type Error = Error<T>;
    type Output = CombinedConfidenceInterval<T>;
    const NAME: &'static str = "Probabilistic Bisector Algorithm";

    fn initialise(
        &mut self,
        problem: &mut Problem<O>,
        mut state: BisectorState<T>,
    ) -> Result<BisectorState<T>, Self::Error> {
        let unscaled_domain = self.scaler.unscaled_domain();

        let slope_sign =
            problem.slope_sign(&unscaled_domain, self.confidence_level, self.max_iter)?;
        state.initialize(
            self.domain.clone(),
            self.n_max,
            self.max_iter,
            slope_sign,
            self.confidence_level,
            self.significance,
        )?;
        Ok(state)
    }

    fn next(
        &mut self,
        problem: &mut Problem<O>,
        mut state: BisectorState<T>,
    ) -> Result<BisectorState<T>, Self::Error> {
        // Can unwrap as the state is necessarily initialised before next can be called
        let distribution = state.distribution.as_mut().unwrap();

        // Generate the next sample point at the median of the current posterior distribution and
        // try to calculate the sign of the sample.
        let next_sample = distribution.median();
        // Translate to real-space: We store in a scaled log space
        let unscaled_sample = self.scaler.unscale_sample(next_sample);

        let try_sign_guess = problem.sign(unscaled_sample, self.confidence_level, self.n_max);

        // If we fail to guess the sign it means we are trying to evaluate in a region which is
        // simply too noisy to be able to determine the sign. This implies the target width
        // requested by the caller is too small. It is not a failure of the algorithm, we can't
        // know a priori how small the confidence interval should be. In this scenario we terminate
        // early and return the current confidence interval.
        //
        // // TODO: This is actually the convergence criterion and we should only terminate when we
        // // hit it. Remove all the target width stuff.
        // if try_sign_guess.as_ref().is_err() {
        //     println!("terminating as too close to root for further improvement");
        //     return Ok(state.terminate_due_to(Reason::Converged));
        // }

        // Unwrap as only one error type could be returned, and it was handled above.
        let sign_guess = try_sign_guess.unwrap();

        // Insert into the posterior distribution
        let slope = state.slope.unwrap();
        if distribution
            .insert(next_sample, sign_guess, slope, self.confidence_level)
            .is_err()
        {
            tracing::error!("terminating as the insertion points are essentially overlapping points currently in the distribution");
            let confidence_levels = state.confidence.unwrap();
            let mut confidence = confidence_levels.last().unwrap();
            confidence.transform(&self.scaler);

            return Err(Error::OverlappingSamples(confidence));
        }

        // Update the confidence interval;
        let confidence_levels = state.confidence.as_mut().unwrap();

        if confidence_levels.update(distribution).is_err() {
            tracing::error!("Failed to update confidence level.
                Typically this means that the algorithm is focussed on an interval which is narrower than can be resolved.
                The distribution gets multiple peaks, and the convex hull used to update the confidence intervals vanishes.");

            let confidence_levels = state.confidence.unwrap();
            let mut confidence = confidence_levels.last().unwrap();
            confidence.transform(&self.scaler);

            return Err(Error::EmptyHull(confidence));
        }

        // Set the new width on the state variable
        state.current_width = state
            .confidence
            .as_ref()
            .map(|confidence| confidence.last_width())
            .unwrap();

        Ok(state)
    }

    fn finalise(
        &mut self,
        _problem: &mut Problem<O>,
        state: BisectorState<T>,
    ) -> Result<Self::Output, Self::Error> {
        let confidence_levels = state.confidence.unwrap();
        let mut confidence = confidence_levels.last().unwrap();
        confidence.transform(&self.scaler);
        Ok(confidence)
    }
}

#[cfg(test)]
mod tests {
    // struct TestFunction {
    //     dist: Normal,
    // }
    //
    // impl TestFunction {
    //     fn new(std_dev: f64) -> Self {
    //         TestFunction {
    //             dist: Normal::new(0.0, std_dev).unwrap(),
    //         }
    //     }
    // }
    //
    // impl Bisectable<f64> for TestFunction {
    //     fn evaluate(&self, x: f64) -> f64 {
    //         let result = (x - 5.0) + self.dist.sample(&mut rand::thread_rng());
    //         dbg!(&result, &x);
    //         result
    //     }
    // }
    // //
    // // #[tracing_test::traced_test]
    // #[test]
    // fn bisect_test() {
    //     let f = TestFunction::new(0.000001);
    //     let domain = 1e-3..10.0;
    //     let bisector = ProbabilisticBisector::new(domain, ConfidenceLevel::ninety_five_percent());
    //
    //     let runner = bisector
    //         .build_for(f)
    //         .configure(|state| state.max_iters(100))
    //         .finalise()
    //         .unwrap();
    //
    //     let result = runner.run();
    //     if let Err(e) = result.as_ref() {
    //         eprintln!("{e:?}");
    //     }
    //     assert!(result.is_ok());
    //     let confidence_interval = result.unwrap();
    //
    //     // dbg!(confidence_interval);
    //     // let fun = |x| {
    //     //     x - Normal::new(0.55, std_dev)
    //     //         .unwrap()
    //     //         .sample(&mut rand::thread_rng())
    //     // };
    //     // let runner = bisect(fun, 0.0, 1.0, ConfidenceLevel::NinetyNinePointNine);
    //     // let result = runner.find();
    //     //
    //     // dbg!(result);
    // }

    #[test]
    fn scaler_with_positive_domain_reconstructs() {
        use super::Scaler;
        let raw = 1.0..10.0;
        let scaler = Scaler::new(raw.clone());

        let unscaled = scaler.unscaled_domain();

        approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
        approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
    }

    #[test]
    fn scaler_with_log_positive_domain_reconstructs() {
        use super::Scaler;
        let raw = 1.0..1e6;
        let scaler = Scaler::new(raw.clone());

        let unscaled = scaler.unscaled_domain();

        approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
        approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
    }

    #[test]
    fn scaler_with_negative_domain_reconstructs() {
        use super::Scaler;
        let raw = -10.0..1.0;
        let scaler = Scaler::new(raw.clone());

        let unscaled = scaler.unscaled_domain();

        approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
        approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
    }

    #[test]
    fn scaler_with_log_negative_domain_reconstructs() {
        use super::Scaler;
        let raw = -1e6..1.0;
        let scaler = Scaler::new(raw.clone());

        let unscaled = scaler.unscaled_domain();

        approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
        approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
    }

    #[test]
    fn scaler_with_domain_spanning_origin_reconstructs() {
        use super::Scaler;
        let raw = -10.0..10.0;
        let scaler = Scaler::new(raw.clone());

        let unscaled = scaler.unscaled_domain();

        approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
        approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
    }
}
