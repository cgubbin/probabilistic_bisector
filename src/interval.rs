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

use super::Distribution;
use confi::{Confidence, ConfidenceInterval, ConfidenceLevel, SignificanceLevel};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::ops::RangeInclusive;

// Confidence intervals over the course of a PBA execution
#[derive(Debug)]
pub struct ConfidenceIntervals<T> {
    // The current number of iterations
    n: usize,
    // The maximum number of iterations
    _n_max: usize,
    // The level of confidence
    confidence_level: ConfidenceLevel<T>,
    // The level of significance used in hypothesis testing
    significance: SignificanceLevel<T>,
    // Confidence intervals seen in iteration order
    values: Vec<CombinedConfidenceInterval<T>>,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ConfidenceError {
    #[error(
        "the convex hull used for the update is empty\n
            this means the algorithm has probably converged to the smallest reasonable value"
    )]
    EmptyHull,
}

impl<T> ConfidenceIntervals<T> {
    pub(super) fn new(
        n_max: usize,
        confidence_level: ConfidenceLevel<T>,
        significance: SignificanceLevel<T>,
    ) -> Self {
        Self {
            n: 0,
            _n_max: n_max,
            confidence_level,
            significance,
            values: Vec::with_capacity(n_max),
        }
    }

    pub(super) fn last(&self) -> Option<CombinedConfidenceInterval<T>>
    where
        T: Float,
    {
        self.values.last().cloned()
    }

    pub(super) fn last_width(&self) -> T
    where
        T: Float + FromPrimitive,
    {
        self.values[self.values.len() - 1].interval.width()
    }

    // pub(super) fn last_width_seq(&self) -> T
    // where
    //     T: Float + FromPrimitive,
    // {
    //     self.values[self.values.len() - 1].seq.width()
    // }

    // fn width(&self, n: usize) -> T
    // where
    //     T: Float + FromPrimitive,
    // {
    //     if n >= self.values.len() {
    //         panic!("n out of bounds");
    //     }
    //     self.values[n].interval.width()
    // }
    //
    // fn seq_width(&self, n: usize) -> T
    // where
    //     T: Float + FromPrimitive,
    // {
    //     if n >= self.values.len() {
    //         panic!("n out of bounds");
    //     }
    //     self.values[n].seq.width()
    // }

    pub(super) fn update(&mut self, posterior: &Distribution<T>) -> Result<(), ConfidenceError>
    where
        T: Float + FromPrimitive + std::fmt::Debug,
    {
        let alpha = self.significance.probability();
        let n1 = T::from_usize(self.n + 1).unwrap();

        let confidence_level = self.confidence_level.probability();
        let one_minus_confidence_level = T::one() - confidence_level;

        let two = T::one() + T::one();
        let d = confidence_level * (two * confidence_level).ln()
            + one_minus_confidence_level * (two * one_minus_confidence_level).ln();

        let beta = (confidence_level / one_minus_confidence_level).ln();

        let next_interval = self.update_confidence(posterior, n1, alpha, beta, d);
        let next_sequential = self.update_sequential(posterior, n1, alpha, beta, d);

        self.n += 1;

        self.values.push(CombinedConfidenceInterval {
            interval: next_interval?,
            seq: next_sequential?,
        });
        Ok(())
    }

    fn update_confidence(
        &self,
        posterior: &Distribution<T>,
        n1: T,
        alpha: T,
        beta: T,
        d: T,
    ) -> Result<ConfidenceInterval<T>, ConfidenceError>
    where
        T: Float + FromPrimitive,
    {
        let two = T::one() + T::one();
        let half = T::one() / two;

        // Eq 3.7
        let b = n1 * d - n1.sqrt() * (-half * (alpha / two).ln()).sqrt() * beta;

        // This is the set G_n defined above equation 3.8
        let g = posterior
            .log_probability_density
            .iter()
            .enumerate()
            .filter(|(_, log_p)| **log_p > b)
            .map(|(j, _)| j)
            .collect::<Vec<_>>();

        // dbg!(&posterior.log_probability_density);
        // std::thread::sleep(std::time::Duration::from_millis(100));

        let convex_hull_endpoints = g.first().ok_or(ConfidenceError::EmptyHull)?
            ..g.last().ok_or(ConfidenceError::EmptyHull)?;

        // The confidence interval is the convex hull of the set g
        Ok(ConfidenceInterval::new(
            posterior.samples[*convex_hull_endpoints.start]
                ..=posterior.samples[convex_hull_endpoints.end + 1], // +1 as there are n + 1 samples, anv we
            // want the last one
            self.confidence_level,
        ))
    }

    fn update_sequential(
        &self,
        posterior: &Distribution<T>,
        n1: T,
        alpha: T,
        beta: T,
        d: T,
    ) -> Result<SequentialConfidenceInterval<T>, ConfidenceError>
    where
        T: Float + FromPrimitive,
    {
        let two = T::one() + T::one();
        let half = T::one() / two;

        // Eq 3.17
        let a = n1 * d - n1.sqrt() * (-half * (alpha / (n1 + T::one())).ln()).sqrt() * beta;

        // The set Ln defined below Eq 3.17
        let l = posterior
            .log_probability_density
            .iter()
            .enumerate()
            .filter(|(_, log_p)| **log_p > a)
            .map(|(j, _)| j)
            .collect::<Vec<_>>();

        // The sequential confidence interval is the convex hull of set l
        let convex_hull_endpoints = l.first().ok_or(ConfidenceError::EmptyHull)?
            ..l.last().ok_or(ConfidenceError::EmptyHull)?;
        let mut interval = posterior.samples[*convex_hull_endpoints.start]
            ..=posterior.samples[convex_hull_endpoints.end + 1];

        // The sequence of intervals is guaranteed to decrease, so the update is only accepted if
        // the calculated confidence interval is narrower that the previous sequential window
        if self.n > 0 {
            interval = (interval.start().max(*self.values[self.n - 1].seq.start()))
                ..=(interval.end().min(*self.values[self.n - 1].seq.end()));
        }

        Ok(SequentialConfidenceInterval::new(
            interval,
            self.confidence_level,
        ))
    }
}

#[derive(Debug, Clone)]
// A confidence interval which is guaranteed to decrease as the algorithm proceeds.
struct SequentialConfidenceInterval<T>(ConfidenceInterval<T>);

impl<T: Float + FromPrimitive> Confidence<T> for SequentialConfidenceInterval<T> {
    fn new(range: RangeInclusive<T>, confidence_level: ConfidenceLevel<T>) -> Self {
        Self(ConfidenceInterval::new(range, confidence_level))
    }

    fn start(&self) -> &T {
        self.0.start()
    }

    fn end(&self) -> &T {
        self.0.end()
    }

    fn confidence_level(&self) -> ConfidenceLevel<T> {
        self.0.confidence_level()
    }
}

impl<T> SequentialConfidenceInterval<T> {
    fn to_f64(self) -> Option<SequentialConfidenceInterval<f64>>
    where
        T: ToPrimitive,
    {
        self.0
            .to_f64()
            .map(|inner| SequentialConfidenceInterval(inner))
    }
}

#[derive(Debug, Clone)]
// Confidence intervals represent the range of values expected to enclose the true value to a
// specified confidence level.
pub struct CombinedConfidenceInterval<T> {
    // The confidence interval evaluated at a given iteration
    pub interval: ConfidenceInterval<T>,
    // The sequential confidence interval at the iteration: sequential confidence intervals are
    // guaranteed to decrease as the algorithm proceeds. No such guarantee is available for the
    // bare interval.
    seq: SequentialConfidenceInterval<T>,
}

impl<T: Float + FromPrimitive> CombinedConfidenceInterval<T> {
    pub(crate) fn transform(&mut self, scaler: &crate::Scaler<T>) {
        self.interval = ConfidenceInterval::new(
            scaler.unscale_sample(*self.interval.start())
                ..=scaler.unscale_sample(*self.interval.end()),
            self.interval.confidence_level(),
        );
        self.seq = SequentialConfidenceInterval(ConfidenceInterval::new(
            scaler.unscale_sample(*self.seq.0.start())..=scaler.unscale_sample(*self.seq.0.end()),
            self.seq.0.confidence_level(),
        ));
    }
}

impl<T> CombinedConfidenceInterval<T> {
    pub(crate) fn to_f64(self) -> Option<CombinedConfidenceInterval<f64>>
    where
        T: ToPrimitive,
    {
        // This should be an if-let chain...
        match (self.interval.to_f64(), self.seq.to_f64()) {
            (Some(interval), Some(seq)) => Some(CombinedConfidenceInterval { interval, seq }),
            _ => None,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::{super::Sign, Certainty, ConfidenceIntervals, Distribution};
//
//     #[test]
//     fn confidence_interval_updates_correctly() {
//         let domain = 0.0..10.0;
//         let mut posterior_distribution = Distribution::new(domain.clone(), 100).unwrap();
//         let mut confidence_intervals = ConfidenceIntervals::new(100);
//         let confidence_level = crate::new::ConfidenceLevel::Ninety;
//         let alpha = Certainty::new(0.05).unwrap();
//
//         // Values increase with the argument
//         let slope = Sign::Positive;
//
//         let true_value = 2.9;
//
//         for i in 0..100 {
//             let median = posterior_distribution.median();
//
//             let sign = if median < true_value {
//                 Sign::Negative
//             } else {
//                 Sign::Positive
//             };
//
//             posterior_distribution.insert(median, sign, slope, confidence_level);
//             confidence_intervals.update(&posterior_distribution, confidence_level, alpha);
//             dbg!(&posterior_distribution, &confidence_intervals);
//         }
//     }
// }
