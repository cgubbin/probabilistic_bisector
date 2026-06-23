// In the PBA we cannot be certain of the true location of the root. Instead we track a
// probability distribution which represents the sum of our knowledge about it's best location.
//
// The `distribution` module contains an implementation of this distribution. The class is initialised
// with a range of values which can be expected to contain the root with a probability approaching
// unity. It stores the range limits, or samples, and the logarithm of the probability in the
// intermediate region ln(1) = 0. As the algorithm proceeds, new sample points are inserted and
// the probabilities are updated.
mod distribution;

mod evolution;

// // The result of a probabilistic bisection is a confidence interval. This expresses a statistical
// // certainty about the range of values enclosing the true value of the root of the objective
// // function.
// //
// // This module contains methods to compute and update confidence intervals as described in §3.3
// mod interval;
// mod intervals;
mod error;
mod root;
mod scaling;
pub(crate) mod semimeet;
pub(crate) mod support;

pub use confi::ConfidenceLevel;
pub use root::{RootError, RootOracle, Sign};

use confi::SignificanceLevel;
use distribution::{PosteriorDistribution, PosteriorError, PosteriorValidationError};
pub use error::PBError;

use scaling::{Scaler, ScalerError};
use semimeet::{Interval, MeetSemiLattice, SemiMeetError, SequentialInterval};
use support::SupportSet;
// pub use interval::CombinedConfidenceInterval;
// use interval::ConfidenceIntervals;
use num_traits::{Float, FromPrimitive};
use std::{fmt, iter, ops::Range};
// use trellis_runner::{Calculation, Problem, TrellisFloat, UserState};

// pub use confi::{Confidence, ConfidenceLevel};
// pub use trellis_runner::GenerateBuilder;

pub struct ProbabilisticBisector<T> {
    // The scaler for the system.
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
}

impl<T: Float + FromPrimitive + std::fmt::Debug> ProbabilisticBisector<T> {
    pub fn new(domain: Range<T>, confidence_level: ConfidenceLevel<T>) -> Result<Self, PBError<T>> {
        let scaler = Scaler::unit_domain_transform(domain.clone())?;

        Ok(Self {
            scaler,
            n_max: 1000,
            max_iter: 100,
            confidence_level,
        })
    }

    fn domain(&self) -> &Range<T> {
        self.scaler.scaled_domain()
    }
}

// pub struct BisectorState<T> {
//     confidence: Option<ConfidenceIntervals<T>>,
//     distribution: Option<Distribution<T>>,
//     slope: Option<Sign>,
//     best_width: T,
//     prev_best_width: T,
//     current_width: T,
// }

// impl<T> BisectorState<T> {
//     fn initialize(
//         &mut self,
//         domain: Range<T>,
//         n_max: usize,
//         max_iter: usize,
//         slope: Sign,
//         confidence_level: ConfidenceLevel<T>,
//         significance_level: SignificanceLevel<T>,
//     ) -> Result<(), Error>
//     where
//         T: Float + std::fmt::Debug,
//     {
//         self.confidence = Some(ConfidenceIntervals::new(
//             max_iter,
//             confidence_level,
//             significance_level,
//         ));

//         self.distribution = Some(Distribution::new(domain, n_max)?);
//         self.slope = Some(slope);
//         Ok(())
//     }
// }

// impl<T> UserState for BisectorState<T>
// where
//     T: Float + FromPrimitive + TrellisFloat + fmt::Debug,
// {
//     type Float = T;
//     type Param = ConfidenceIntervals<T>;

//     fn new() -> Self {
//         Self {
//             confidence: None,
//             distribution: None,
//             slope: None,
//             best_width: T::max_value(),
//             prev_best_width: T::max_value(),
//             current_width: T::max_value(),
//         }
//     }

//     fn update(&mut self) -> trellis_runner::ErrorEstimate<Self::Float> {
//         trellis_runner::ErrorEstimate(self.current_width)
//     }

//     fn is_initialised(&self) -> bool {
//         self.confidence.is_some() && self.slope.is_some() && self.distribution.is_some()
//     }

//     fn get_param(&self) -> Option<&Self::Param> {
//         self.confidence.as_ref()
//     }

//     fn last_was_best(&mut self) {
//         std::mem::swap(&mut self.prev_best_width, &mut self.best_width);
//         self.best_width = self.current_width;
//     }
// }

// impl<O, T> Bisectable<T> for Problem<O>
// where
//     T: Float + FromPrimitive + std::fmt::Debug,
//     O: Bisectable<T>,
// {
//     fn evaluate(&self, x: T) -> T {
//         self.as_ref().evaluate(x)
//     }
// }

// impl<O, T> Calculation<O, BisectorState<T>> for ProbabilisticBisector<T>
// where
//     T: Float + FromPrimitive + fmt::Debug + iter::Sum + TrellisFloat,
//     O: Bisectable<T>,
// {
//     type Error = Error;
//     type Output = CombinedConfidenceInterval<T>;
//     const NAME: &'static str = "Probabilistic Bisector Algorithm";

//     fn initialise(
//         &mut self,
//         problem: &mut Problem<O>,
//         mut state: BisectorState<T>,
//     ) -> Result<BisectorState<T>, Self::Error> {
//         let unscaled_domain = self.scaler.unscaled_domain();

//         let slope_sign =
//             problem.slope_sign(&unscaled_domain, self.confidence_level, self.max_iter)?;
//         state.initialize(
//             self.domain.clone(),
//             self.n_max,
//             self.max_iter,
//             slope_sign,
//             self.confidence_level,
//             self.significance,
//         )?;
//         Ok(state)
//     }

//     fn next(
//         &mut self,
//         problem: &mut Problem<O>,
//         mut state: BisectorState<T>,
//     ) -> Result<BisectorState<T>, Self::Error> {
//         // Can unwrap as the state is necessarily initialised before next can be called
//         let distribution = state.distribution.as_mut().unwrap();

//         // Generate the next sample point at the median of the current posterior distribution and
//         // try to calculate the sign of the sample.
//         let next_sample = distribution.median();
//         // Translate to real-space: We store in a scaled log space
//         let unscaled_sample = self.scaler.unscale_sample(next_sample);

//         let try_sign_guess = problem.sign(unscaled_sample, self.confidence_level, self.n_max);

//         // If we fail to guess the sign it means we are trying to evaluate in a region which is
//         // simply too noisy to be able to determine the sign. This implies the target width
//         // requested by the caller is too small. It is not a failure of the algorithm, we can't
//         // know a priori how small the confidence interval should be. In this scenario we terminate
//         // early and return the current confidence interval.
//         //
//         // // TODO: This is actually the convergence criterion and we should only terminate when we
//         // // hit it. Remove all the target width stuff.
//         // if try_sign_guess.as_ref().is_err() {
//         //     println!("terminating as too close to root for further improvement");
//         //     return Ok(state.terminate_due_to(Reason::Converged));
//         // }

//         // Unwrap as only one error type could be returned, and it was handled above.
//         let sign_guess = try_sign_guess.unwrap();

//         // Insert into the posterior distribution
//         let slope = state.slope.unwrap();
//         if distribution
//             .insert(next_sample, sign_guess, slope, self.confidence_level)
//             .is_err()
//         {
//             tracing::error!("terminating as the insertion points are essentially overlapping points currently in the distribution");
//             let confidence_levels = state.confidence.unwrap();
//             let mut confidence = confidence_levels.last().unwrap();
//             confidence.transform(&self.scaler);

//             return Err(Error::OverlappingSamples(confidence.to_f64().unwrap()));
//         }

//         // Update the confidence interval;
//         let confidence_levels = state.confidence.as_mut().unwrap();

//         if confidence_levels.update(distribution).is_err() {
//             tracing::error!("Failed to update confidence level.
//                 Typically this means that the algorithm is focussed on an interval which is narrower than can be resolved.
//                 The distribution gets multiple peaks, and the convex hull used to update the confidence intervals vanishes.");

//             let confidence_levels = state.confidence.unwrap();
//             let mut confidence = confidence_levels.last().unwrap();
//             confidence.transform(&self.scaler);

//             return Err(Error::EmptyHull(confidence.to_f64().unwrap()));
//         }

//         // Set the new width on the state variable
//         state.current_width = state
//             .confidence
//             .as_ref()
//             .map(|confidence| confidence.last_width())
//             .unwrap();

//         Ok(state)
//     }

//     fn finalise(
//         &mut self,
//         _problem: &mut Problem<O>,
//         state: BisectorState<T>,
//     ) -> Result<Self::Output, Self::Error> {
//         let confidence_levels = state.confidence.unwrap();
//         let mut confidence = confidence_levels.last().unwrap();
//         confidence.transform(&self.scaler);
//         Ok(confidence)
//     }
// }

// #[cfg(test)]
// mod tests {
//     // struct TestFunction {
//     //     dist: Normal,
//     // }
//     //
//     // impl TestFunction {
//     //     fn new(std_dev: f64) -> Self {
//     //         TestFunction {
//     //             dist: Normal::new(0.0, std_dev).unwrap(),
//     //         }
//     //     }
//     // }
//     //
//     // impl Bisectable<f64> for TestFunction {
//     //     fn evaluate(&self, x: f64) -> f64 {
//     //         let result = (x - 5.0) + self.dist.sample(&mut rand::thread_rng());
//     //         dbg!(&result, &x);
//     //         result
//     //     }
//     // }
//     // //
//     // // #[tracing_test::traced_test]
//     // #[test]
//     // fn bisect_test() {
//     //     let f = TestFunction::new(0.000001);
//     //     let domain = 1e-3..10.0;
//     //     let bisector = ProbabilisticBisector::new(domain, ConfidenceLevel::ninety_five_percent());
//     //
//     //     let runner = bisector
//     //         .build_for(f)
//     //         .configure(|state| state.max_iters(100))
//     //         .finalise()
//     //         .unwrap();
//     //
//     //     let result = runner.run();
//     //     if let Err(e) = result.as_ref() {
//     //         eprintln!("{e:?}");
//     //     }
//     //     assert!(result.is_ok());
//     //     let confidence_interval = result.unwrap();
//     //
//     //     // dbg!(confidence_interval);
//     //     // let fun = |x| {
//     //     //     x - Normal::new(0.55, std_dev)
//     //     //         .unwrap()
//     //     //         .sample(&mut rand::thread_rng())
//     //     // };
//     //     // let runner = bisect(fun, 0.0, 1.0, ConfidenceLevel::NinetyNinePointNine);
//     //     // let result = runner.find();
//     //     //
//     //     // dbg!(result);
//     // }

//     #[test]
//     fn scaler_with_positive_domain_reconstructs() {
//         use super::Scaler;
//         let raw = 1.0..10.0;
//         let scaler = Scaler::new(raw.clone());

//         let unscaled = scaler.unscaled_domain();

//         approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
//         approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
//     }

//     #[test]
//     fn scaler_with_log_positive_domain_reconstructs() {
//         use super::Scaler;
//         let raw = 1.0..1e6;
//         let scaler = Scaler::new(raw.clone());

//         let unscaled = scaler.unscaled_domain();

//         approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
//         approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
//     }

//     #[test]
//     fn scaler_with_negative_domain_reconstructs() {
//         use super::Scaler;
//         let raw = -10.0..1.0;
//         let scaler = Scaler::new(raw.clone());

//         let unscaled = scaler.unscaled_domain();

//         approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
//         approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
//     }

//     #[test]
//     fn scaler_with_log_negative_domain_reconstructs() {
//         use super::Scaler;
//         let raw = -1e6..1.0;
//         let scaler = Scaler::new(raw.clone());

//         let unscaled = scaler.unscaled_domain();

//         approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
//         approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
//     }

//     #[test]
//     fn scaler_with_domain_spanning_origin_reconstructs() {
//         use super::Scaler;
//         let raw = -10.0..10.0;
//         let scaler = Scaler::new(raw.clone());

//         let unscaled = scaler.unscaled_domain();

//         approx::assert_relative_eq!(raw.start, unscaled.start, epsilon = 1e-10);
//         approx::assert_relative_eq!(raw.end, unscaled.end, epsilon = 1e-10);
//     }
// }
