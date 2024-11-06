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

use confidence::{ConfidenceLevel, Scale, SignificanceLevel};
use distribution::{Distribution, DistributionError};
pub(crate) use interval::CombinedConfidenceInterval;
use interval::ConfidenceIntervals;
use num_traits::{Float, FromPrimitive};
use std::{fmt, iter, ops::Range};
use trellis_runner::{Calculation, Problem, TrellisFloat, UserState};

#[derive(Debug, thiserror::Error)]
pub enum ProbabalisticBisectorError {
    #[error("error in distribution computation: {0}")]
    DistributionError(#[from] DistributionError),
    #[error("failed to determine the function sign at {0} in less than {1} iterations")]
    SignDeterminationError(f64, usize),
    #[error("in computing the slope of the function, no root was detected in the domain")]
    NoRootDetected,
}

// Representing the sign of a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sign {
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
pub(crate) trait Bisectable<T: Float + FromPrimitive + std::fmt::Debug> {
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
    ) -> Result<Sign, ProbabalisticBisectorError> {
        let sign_start = self.sign(domain.start, confidence_level, max_iter)?;
        let sign_end = self.sign(domain.end, confidence_level, max_iter)?;

        match (sign_start, sign_end) {
            (Sign::Positive, Sign::Negative) => Ok(Sign::Negative),
            (Sign::Negative, Sign::Positive) => Ok(Sign::Positive),
            _ => Err(ProbabalisticBisectorError::NoRootDetected),
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
    ) -> Result<Sign, ProbabalisticBisectorError> {
        let mut random_walk = T::zero();
        let p_c = confidence_level.probability();
        let two = T::one() + T::one();
        let one = T::one();

        for ii in 0..max_iter {
            random_walk = random_walk + self.evaluate(x).signum();
            let n = T::from_usize(ii).unwrap();

            let power_test = ((two * n) * ((n + one).ln() - two.ln() - p_c.ln())).sqrt();

            dbg!(&random_walk, &power_test);
            if random_walk.abs() > power_test {
                return Ok(random_walk.into());
            }
        }

        Err(ProbabalisticBisectorError::SignDeterminationError(
            x.to_f64().unwrap(),
            max_iter,
        ))
    }
}

pub(crate) struct ProbabalisticBisector<T> {
    // The range of values expected with high certainty to contain the true value of the root
    //
    // When creating this struct we transform the domain into logarithmic space, scaled on [, 1].
    // This improves convergence and ensures robust behaviour for inputs which vary over many
    // orders of magnitude
    domain: Range<T>,
    // The scaling factor for the domain
    //
    // The logarithmic domain is scaled to [0, 1] to improve convergence.
    scaling_factor: T,
    shift: T,
    // The maximum number of elements to use in computing the underlying probability distribution
    n_max: usize,
    // The maximum number of iterations
    max_iter: usize,
    // The confidence level
    confidence_level: ConfidenceLevel<T>,
    significance: SignificanceLevel<T>,
    target_width: T,
}

impl<T: Float + FromPrimitive + std::fmt::Debug> ProbabalisticBisector<T> {
    pub(crate) fn new(
        domain: Range<T>,
        confidence_level: ConfidenceLevel<T>,
        target_width: T,
    ) -> Self {
        // TODO: Original impl set the significance to double this value. Does this make sense?
        let significance: SignificanceLevel<T> = confidence_level.into();

        // Transform the domain into logarithmic space
        let log_10_scaled_domain = domain.start.log10()..domain.end.log10();
        let scaling_factor =
            (log_10_scaled_domain.end - log_10_scaled_domain.start) / T::from_f64(2.0).unwrap();
        let shift = -T::one() - log_10_scaled_domain.start / scaling_factor;

        // let domain = (log_10_scaled_domain.start / scaling_factor + shift)
        //     ..(log_10_scaled_domain.end / scaling_factor + shift);
        //
        // let actual = (domain.start - shift) * scaling_factor..(domain.end - shift) * scaling_factor;
        // dbg!(&domain, &actual);
        //
        // panic!();
        // let scaling_factor = log_10_scaled_domain.end;
        // let domain = (log_10_scaled_domain.start / scaling_factor)..T::one();
        let domain = -T::one()..T::one();

        Self {
            domain,
            scaling_factor,
            shift,
            n_max: 1000,
            max_iter: 100,
            confidence_level,
            significance,
            target_width,
        }
    }

    fn shift_sample(&self, sample: T) -> T {
        (sample - self.shift) * self.scaling_factor
    }

    fn unscale_sample(&self, sample: T) -> T {
        T::from_f64(10.0).unwrap().powf(self.shift_sample(sample))

        // T::from_f64(10.0)
        //     .unwrap()
        //     .powf(self.scaling_factor * sample)
    }

    fn unscaled_domain(&self, domain: Range<T>) -> Range<T> {
        self.unscale_sample(domain.start)..self.unscale_sample(domain.end)
    }
}

pub(crate) struct BisectorState<T> {
    confidence: Option<ConfidenceIntervals<T>>,
    distribution: Option<Distribution<T>>,
    slope: Option<Sign>,
    best_width: T,
    prev_best_width: T,
    current_width: T,
    target_width: T,
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
        target_width: T,
    ) -> Result<(), ProbabalisticBisectorError>
    where
        T: Float + std::fmt::Debug,
    {
        self.confidence = Some(ConfidenceIntervals::new(
            max_iter,
            confidence_level,
            Scale::Log10,
            significance_level,
        ));

        self.distribution = Some(Distribution::new(domain, n_max)?);
        self.slope = Some(slope);
        self.target_width = target_width;
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
            target_width: T::zero(),
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

impl<O, T> Calculation<O, BisectorState<T>> for ProbabalisticBisector<T>
where
    T: Float + FromPrimitive + fmt::Debug + iter::Sum + TrellisFloat,
    O: Bisectable<T>,
{
    type Error = ProbabalisticBisectorError;
    type Output = CombinedConfidenceInterval<T>;
    const NAME: &'static str = "Probabalistic Bisector Algorithm";

    fn initialise(
        &mut self,
        problem: &mut Problem<O>,
        mut state: BisectorState<T>,
    ) -> Result<BisectorState<T>, Self::Error> {
        let unscaled_domain = self.unscaled_domain(self.domain.clone());

        let slope_sign =
            problem.slope_sign(&unscaled_domain, self.confidence_level, self.max_iter)?;
        state.initialize(
            self.domain.clone(),
            self.n_max,
            self.max_iter,
            slope_sign,
            self.confidence_level,
            self.significance,
            self.target_width,
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
        let unscaled_sample = self.unscale_sample(next_sample);

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
            panic!();
            // return Ok(state.terminate_due_to(Reason::Converged));
        }

        // Update the confidence interval;
        let confidence_levels = state.confidence.as_mut().unwrap();

        if confidence_levels.update(distribution).is_err() {
            tracing::error!("Failed to update confidence level.
                Typically this means that the algorithm is focussed on an interval which is narrower than can be resolved.
                The distribution gets multiple peaks, and the convex hull used to update the confidence intervals vanishes.");

            panic!();
            // return Ok(state.terminate_due_to(Reason::Converged));
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
        // Roll back the scaling
        confidence.scaled(self.scaling_factor, self.shift);
        // Return after converting to linear space
        Ok(confidence.linear())
    }
}

#[cfg(test)]
mod tests {
    use rand::distributions::Distribution;
    use statrs::distribution::Normal;
    use trellis_runner::GenerateBuilder;

    use super::*;

    struct TestFunction {
        dist: Normal,
    }

    impl TestFunction {
        fn new(std_dev: f64) -> Self {
            TestFunction {
                dist: Normal::new(0.0, std_dev).unwrap(),
            }
        }
    }

    impl Bisectable<f64> for TestFunction {
        fn evaluate(&self, x: f64) -> f64 {
            let result = (x - 5.0) + self.dist.sample(&mut rand::thread_rng());
            dbg!(&result, &x);
            result
        }
    }

    #[tracing_test::traced_test]
    #[test]
    fn bisect_test() {
        let f = TestFunction::new(0.000001);
        let domain = 1e-3..10.0;
        let bisector =
            ProbabalisticBisector::new(domain, ConfidenceLevel::ninety_five_percent(), 1e-6);

        let runner = bisector
            .build_for(f)
            .configure(|state| state.max_iters(100))
            .finalise()
            .unwrap();

        let result = runner.run();
        if let Err(e) = result.as_ref() {
            eprintln!("{e:?}");
        }
        assert!(result.is_ok());
        let confidence_interval = result.unwrap();

        // dbg!(confidence_interval);
        // let fun = |x| {
        //     x - Normal::new(0.55, std_dev)
        //         .unwrap()
        //         .sample(&mut rand::thread_rng())
        // };
        // let runner = bisect(fun, 0.0, 1.0, ConfidenceLevel::NinetyNinePointNine);
        // let result = runner.find();
        //
        // dbg!(result);
    }
}
