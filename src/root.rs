//! # Root-Finding Oracle
//!
//! This module defines the trait implemented by objective functions that can
//! be analysed by the probabilistic bisection algorithm.
//!
//! Implementors provide a single method, [`RootOracle::evaluate`], which
//! evaluates the objective at a supplied point in the domain.
//!
//! Evaluations may be stochastic: repeated calls with the same argument are
//! permitted to return different values.
//!
//! ## Oracle model
//!
//! The probabilistic bisection algorithm never observes the true objective
//! directly. Instead it interacts with an oracle capable of producing noisy
//! evaluations of the objective.
//!
//! From these evaluations the algorithm infers:
//!
//! - the sign of the objective at a query point
//! - the monotone direction of the objective over the domain
//! - the location of a root
//!
//! ## Assumptions
//!
//! The algorithm assumes:
//!
//! - the objective is scalar-valued
//! - the objective is monotone over the search domain
//! - the domain brackets a root
//!
//! These assumptions allow noisy sign observations to be converted into
//! probabilistic information about the root location.
//!
//! ## Example
//!
//! ```
//! use probabilistic_bisector::{RootOracle, Sign, ConfidenceLevel};
//!
//! struct Linear;
//!
//! impl RootOracle<f64> for Linear {
//!     fn evaluate(&mut self, x: f64) -> f64 {
//!         x
//!     }
//! }
//!
//! let confidence_level = ConfidenceLevel::<f64>::CL95;
//! let negative_sign = Linear.sign(-1.0, confidence_level, 100)?;
//! assert_eq!(negative_sign, Sign::Negative);
//! let positive_sign = Linear.sign(1.0, confidence_level, 100)?;
//! assert_eq!(positive_sign, Sign::Positive);
//! let slope_sign = Linear.slope_sign(&(-1.0..1.0), confidence_level, 100)?;
//! assert_eq!(slope_sign, Sign::Positive);
//! # Ok::<_, probabilistic_bisector::RootError>(())
//!
//! ```
use confi::ConfidenceLevel;
use num_traits::{Float, FromPrimitive};
use std::{fmt, ops::Range};

#[derive(thiserror::Error, Debug)]
pub enum RootError {
    #[error("tried to evaluate at a NaN value")]
    NaN,
    #[error("failed to sign to prescribed confidence after {0} iterations")]
    MaxIterExceeded(usize),
    #[error("no root detected in the domain")]
    NoRootDetected,
}

// Representing the sign of a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Positive,
    Negative,
}

impl Sign {
    // Try to construct a sign
    fn try_from<T: Float>(x: T) -> Result<Option<Self>, RootError> {
        let sign = x.signum();

        if sign == T::one() {
            Ok(Some(Sign::Positive))
        } else if sign == -T::one() {
            Ok(Some(Sign::Negative))
        } else {
            return Err(RootError::NaN);
        }
    }

    fn signum<T: Float>(&self) -> T {
        match self {
            Sign::Positive => T::one(),
            Sign::Negative => T::one().neg(),
        }
    }
}

// Trait for bisectable objective functions
pub trait RootOracle<T: Float + FromPrimitive + fmt::Debug> {
    /// Evaluate the objective function at `x`.
    ///
    /// Implementations may be stochastic: repeated calls with the same `x`
    /// may return different values. The returned value is interpreted only
    /// through its sign by [`RootOracle::sign`].
    fn evaluate(&mut self, x: T) -> T;

    // The sign of the slope of the function over the domain
    //
    // It is a necessary condition that the function has a root within the domain. This means it is
    // required that the sign of the function changes between the start and end of the domain. We
    // can therefore determine the slope of the function by evaluating the sign at the start and end
    #[tracing::instrument(ret, level = tracing::Level::DEBUG, skip(self))]
    fn slope_sign(
        &mut self,
        domain: &Range<T>,
        confidence_level: ConfidenceLevel<T>,
        max_iter: usize,
    ) -> Result<Sign, RootError> {
        let sign_start = self.sign(domain.start, confidence_level, max_iter)?;
        let sign_end = self.sign(domain.end, confidence_level, max_iter)?;

        match (sign_start, sign_end) {
            (Sign::Positive, Sign::Negative) => Ok(Sign::Negative),
            (Sign::Negative, Sign::Positive) => Ok(Sign::Positive),
            _ => Err(RootError::NoRootDetected),
        }
    }

    /// Estimate the sign of the objective at `x` to the requested confidence level.
    ///
    /// The objective is assumed to be observable only through noisy evaluations.
    /// This method repeatedly evaluates the objective at `x`, converts each
    /// non-zero observation to its sign, and accumulates the resulting random walk.
    ///
    /// Sampling stops once the curved-boundary test is crossed. If the boundary
    /// is not crossed within `max_iter` evaluations, the method returns
    /// [`RootError::MaxIterExceeded`].
    ///
    /// Exact zero observations are treated as non-informative and do not
    /// contribute to the random walk.
    #[tracing::instrument(ret, level = tracing::Level::DEBUG, skip(self))]
    fn sign(
        &mut self,
        x: T,
        confidence_level: ConfidenceLevel<T>,
        max_iter: usize,
    ) -> Result<Sign, RootError> {
        let mut random_walk = T::zero();
        let alpha = confidence_level.significance().into_inner();
        let two = T::one() + T::one();
        let one = T::one();

        for ii in 0..max_iter {
            let y = self.evaluate(x);
            if y.is_nan() {
                return Err(RootError::NaN);
            }

            if y == T::zero() {
                continue;
            }

            random_walk = random_walk + y.signum();

            let n = T::from_usize(ii + 1).unwrap();

            let power_test = ((two * n) * ((n + one).ln() - two.ln() - alpha.ln())).sqrt();

            if random_walk.abs() > power_test {
                // Random walk cannot be NaN, as an error would have been returned. It must be
                // greater than zero as power_test is a positive number. Therefore the unwrap is
                // safe
                return Sign::try_from(random_walk).map(|sign| sign.unwrap());
            }
        }

        Err(RootError::MaxIterExceeded(max_iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Constant(f64);

    impl RootOracle<f64> for Constant {
        fn evaluate(&mut self, _x: f64) -> f64 {
            self.0
        }
    }

    struct Linear;

    impl RootOracle<f64> for Linear {
        fn evaluate(&mut self, x: f64) -> f64 {
            x
        }
    }

    struct NanObjective;

    impl RootOracle<f64> for NanObjective {
        fn evaluate(&mut self, _x: f64) -> f64 {
            f64::NAN
        }
    }

    struct ZeroObjective;

    impl RootOracle<f64> for ZeroObjective {
        fn evaluate(&mut self, _x: f64) -> f64 {
            0.0
        }
    }

    #[test]
    fn sign_detects_positive_objective() {
        let mut f = Constant(1.0);

        let sign = f
            .sign(0.0, ConfidenceLevel::new(0.8).unwrap(), 100)
            .unwrap();

        assert_eq!(sign, Sign::Positive);
    }

    #[test]
    fn sign_detects_negative_objective() {
        let mut f = Constant(-1.0);

        let sign = f
            .sign(0.0, ConfidenceLevel::new(0.8).unwrap(), 100)
            .unwrap();

        assert_eq!(sign, Sign::Negative);
    }

    #[test]
    fn sign_returns_nan_error_for_nan_evaluation() {
        let mut f = NanObjective;

        let err = f
            .sign(0.0, ConfidenceLevel::new(0.8).unwrap(), 100)
            .unwrap_err();

        assert!(matches!(err, RootError::NaN));
    }

    #[test]
    fn zero_observations_do_not_determine_sign() {
        let mut f = ZeroObjective;

        let err = f
            .sign(0.0, ConfidenceLevel::new(0.8).unwrap(), 10)
            .unwrap_err();

        assert!(matches!(err, RootError::MaxIterExceeded(10)));
    }

    #[test]
    fn sign_can_fail_when_iteration_budget_is_too_small() {
        let mut f = Constant(1.0);

        let err = f
            .sign(0.0, ConfidenceLevel::new(0.99).unwrap(), 1)
            .unwrap_err();

        assert!(matches!(err, RootError::MaxIterExceeded(1)));
    }

    #[test]
    fn slope_sign_detects_positive_slope() {
        let mut f = Linear;

        let slope = f
            .slope_sign(&(-1.0..1.0), ConfidenceLevel::new(0.8).unwrap(), 100)
            .unwrap();

        assert_eq!(slope, Sign::Positive);
    }

    #[test]
    fn slope_sign_detects_negative_slope() {
        struct NegativeLinear;

        impl RootOracle<f64> for NegativeLinear {
            fn evaluate(&mut self, x: f64) -> f64 {
                -x
            }
        }

        let mut f = NegativeLinear;

        let slope = f
            .slope_sign(&(-1.0..1.0), ConfidenceLevel::new(0.8).unwrap(), 100)
            .unwrap();

        assert_eq!(slope, Sign::Negative);
    }

    #[test]
    fn slope_sign_errors_when_no_root_is_bracketed() {
        let mut f = Constant(1.0);

        let err = f
            .slope_sign(&(0.0..1.0), ConfidenceLevel::new(0.8).unwrap(), 100)
            .unwrap_err();

        assert!(matches!(err, RootError::NoRootDetected));
    }
}
