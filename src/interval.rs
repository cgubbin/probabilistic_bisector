//! In a probabilistic bisection algorithm we can only observe the objective function in the
//! presence of noise. This means that, no matter how many observations are made we can never find
//! the value of the root to arbitrary precision.
//!
//! The result of successful execution of the PBA is a confidence interval, representing the range
//! of input values expected to contain the root to a precision specified to the caller. These
//! provide a statistical guarantee to the caller that the true value of the root is within the
//! range.
//!
//! This module provides methods for updating [`ConfidenceIntervals`] as the algorithm executes.
//! The implementation in this module follows that described in §3.3 of Waeber.

use super::PosteriorDistribution;
use confi::{ConfidenceInterval, ConfidenceLevel, SignificanceLevel};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::ops::RangeInclusive;

pub struct ConfidenceSnapshot<T> {
    pub interval: WaeberInterval<T>,
    pub sequential: SequentialInterval<T>,
}

#[derive(Clone, Debug)]
pub struct WaeberInterval<T> {
    lower: T,
    upper: T,
}

impl<T> WaeberInterval<T> {
    fn new(g: &[usize], knots: &[T]) -> Result<Self, ConfidenceError>
    where
        T: Float,
    {
        let start = *g.first().ok_or(ConfidenceError::EmptyHull)?;
        let end = *g.last().ok_or(ConfidenceError::EmptyHull)?;

        Ok(Self {
            lower: knots[start],
            upper: knots[end + 1],
        })
    }
}

#[derive(Clone, Debug)]
pub struct SequentialInterval<T> {
    inner: WaeberInterval<T>,
}

impl<T: Float> SequentialInterval<T> {
    fn new(
        raw: WaeberInterval<T>,
        prev: Option<&SequentialInterval<T>>,
    ) -> Result<Self, ConfidenceError> {
        if let Some(prev) = prev {
            let lower = raw.lower.max(prev.inner.lower);
            let upper = raw.upper.min(prev.inner.upper);

            if raw.upper > prev.inner.upper || raw.lower < prev.inner.upper {
                return Err(ConfidenceError::NonOverlappingSequentialIntervals);
            }

            Ok(Self {
                inner: WaeberInterval { lower, upper },
            })
        } else {
            Ok(Self { inner: raw })
        }
    }
}

pub struct State<T> {
    n: usize,
    max_n: usize,
    confidence_level: ConfidenceLevel<T>,
    significance: SignificanceLevel<T>,
    values: Vec<ConfidenceSnapshot<T>>,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ConfidenceError {
    #[error(
        "the convex hull used for the update is empty\n
            this means the algorithm has probably converged to the smallest reasonable value"
    )]
    EmptyHull,
    #[error(
        "the convex hull used for the update is empty\n
            this means the algorithm has probably converged to the smallest reasonable value"
    )]
    DegenerateWaeberRegion,
    #[error("invalid confidence level or interval constructed: {0}")]
    Confi(#[from] confi::ValidationError),
    #[error("invalid sequential state")]
    NonOverlappingSequentialIntervals,
}

impl<T> State<T> {
    pub(super) fn new(
        max_n: usize,
        confidence_level: ConfidenceLevel<T>,
        significance: SignificanceLevel<T>,
    ) -> Self {
        Self {
            n: 0,
            max_n,
            confidence_level,
            significance,
            values: Vec::with_capacity(max_n),
        }
    }

    pub(super) fn last(&self) -> Option<&ConfidenceSnapshot<T>> {
        self.values.last()
    }

    pub(super) fn update(
        &mut self,
        posterior: &PosteriorDistribution<T>,
    ) -> Result<(), ConfidenceError>
    where
        T: Float + FromPrimitive + std::fmt::Debug,
    {
        let snapshot = self.compute_snapshot(posterior)?;

        self.n += 1;
        self.values.push(snapshot);

        Ok(())
    }

    fn compute_snapshot(
        &self,
        posterior: &PosteriorDistribution<T>,
    ) -> Result<ConfidenceSnapshot<T>, ConfidenceError>
    where
        T: Float + FromPrimitive,
    {
        // ---- Precompute constants used in Waeber-style thresholding ----
        //
        // These constants encode:
        // - confidence level scaling
        // - significance correction
        // - KL-like adjustment term (d)
        // - likelihood ratio shift (beta)
        //
        // They are iteration-dependent only through `n1` and significance scaling.
        let alpha = self.significance.into_inner();
        let n1 = T::from_usize(self.n + 1).unwrap();

        let c = self.confidence_level.into_inner();
        let one_minus_c = T::one() - c;

        let two = T::one() + T::one();

        let d = c * (two * c).ln() + one_minus_c * (two * one_minus_c).ln();

        let beta = (c / one_minus_c).ln();

        // ---- Threshold construction (Eq 3.7 in Waeber-style derivation) ----
        //
        // This threshold selects "high posterior mass intervals" in log-space.
        // Conceptually: keep indices where posterior mass exceeds a dynamic bound.
        let b = n1 * d - n1.sqrt() * (-(T::one() / two) * (alpha / two).ln()).sqrt() * beta;

        let b_shifted = b - posterior.max_log_interval_mass();

        // ---- High-probability index set ----
        //
        // G_n = indices where posterior log-density exceeds threshold b
        //
        // This set defines the support of the confidence interval.
        let g: Vec<usize> = posterior
            .shifted_log_interval_mass()
            .enumerate()
            .filter(|(_, lp)| *lp > b_shifted)
            .map(|(i, _)| i)
            .collect();

        let interval = WaeberInterval::new(&g, &posterior.knots)?;

        let sequential = SequentialInterval::new(
            interval.clone(),
            self.values.last().map(|value| &value.sequential),
        )?;

        Ok(ConfidenceSnapshot {
            interval,
            sequential,
        })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::Sign;
//     use proptest::*;

//     proptest! {
//         #[test]
//         fn confidence_intervals_are_valid_and_monotone(
//             ops in proptest::collection::vec(0f64..1f64, 1..50)
//         ) {
//             let mut posterior = PosteriorDistribution::new(0.0,1.0, 128).unwrap();

//             let mut ci = State::new(
//                 128,
//                 ConfidenceLevel::<f64>::CL95,
//                 SignificanceLevel::<f64>::SL05
//             );

//             let mut prev_seq: Option<(f64, f64)> = None;

//             for x in ops {
//                 let direction = Sign::Positive;

//                 posterior.observe(x, direction, ConfidenceLevel::new(0.95).unwrap()).unwrap();
//                 ci.update(&posterior).unwrap();

//                 let last = ci.last().unwrap();

//                 let (l, r) = (last.sequential.lower(), last.sequential.upper());

//                 // 1. validity
//                 prop_assert!(l <= r);

//                 // 2. monotonic shrinkage
//                 if let Some((pl, pr)) = prev_seq {
//                     prop_assert!(l >= pl);
//                     prop_assert!(r <= pr);
//                 }

//                 prev_seq = Some((l, r));
//             }
//         }
//     }

//     proptest! {
//         #[test]
//         fn intervals_do_not_collapse(
//             ops in proptest::collection::vec(0f64..1f64, 10..50)
//         ) {
//             let mut posterior = PosteriorDistribution::new(0.0, 1.0, 64).unwrap();
//             let mut ci = State::new(
//                 64,
//                 ConfidenceLevel::<f64>::CL95,
//                 SignificanceLevel::<f64>::SL05,
//             );

//             for x in ops {
//                 posterior.observe(x, Sign::Positive, ConfidenceLevel::<f64>::CL95).unwrap();
//                 let _ = ci.update(&posterior).unwrap();
//             }

//             let last = ci.last().unwrap();

//             prop_assert!(last.sequential.lower() < last.sequential.upper());
//         }
//     }

//     proptest! {
//         #[test]
//         fn biased_updates_shift_intervals_right(
//             ops in proptest::collection::vec(0.6f64..1.0f64, 10..50)
//         ) {
//             let mut posterior = PosteriorDistribution::new(0.0, 1.0, 64).unwrap();
//             let mut ci = State::new(
//                 64,
//                 ConfidenceLevel::<f64>::CL95,
//                 SignificanceLevel::<f64>::SL05,
//             );

//             for x in ops {
//                 posterior.observe(x, Sign::Positive, ConfidenceLevel::<f64>::CL95).unwrap();
//                 let _ = ci.update(&posterior).unwrap();
//             }

//             let last = ci.last().unwrap();

//             // weak sanity check: center should drift right
//             prop_assert!(last.sequential.centre() > 0.4);
//         }
//     }
// }
