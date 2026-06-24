//! # Support Set Module
//!
//! This module defines the **active support of a posterior measure**.
//!
//! The support is the subset of intervals whose probability mass
//! exceeds a numerical threshold.
//!
//! ## Conceptual role
//!
//! The support acts as a *projection* of the posterior:
//!
//! ```text
//! posterior → support
//! ```
//!
//! It identifies regions of the domain that remain statistically relevant
//! under the current belief state.
//!
//! ## Mathematical interpretation
//!
//! Let P_i be the probability mass of interval i.
//!
//! The support is:
//!
//! ```text
//! S = ⋃ { I_i | P_i > ε }
//! ```
//!
//! where ε is a fixed numerical tolerance in probability space.
//!
//! ## Properties
//!
//! - Monotone under renormalized updates
//! - Computed deterministically from posterior state
//! - Does not store probability mass itself
//!
//! ## Design principle
//!
//! The support is a *derived structure*, not primary state.
//!
//! It is recomputed from the posterior at each inference step
//! to avoid drift and accumulation of numerical error.
use crate::{Interval, IntervalError, PosteriorDistribution};

use num_traits::{Float, FromPrimitive};

/// A derived representation of the posterior’s active region.
///
/// Contains all intervals whose probability mass exceeds a
/// numerical threshold ε.
///
/// This structure is fully recomputed from the posterior and
/// does not maintain independent state.
#[derive(Clone, Debug)]
pub(crate) struct SupportSet<T> {
    /// Intervals in the domain that carry non-negligible probability mass.
    active_intervals: Vec<Interval<T>>,
}

impl<T> SupportSet<T>
where
    T: Float + FromPrimitive,
{
    /// Create a new support set from a posterior distribution
    pub(crate) fn new(posterior: &PosteriorDistribution<T>) -> Result<Self, IntervalError<T>> {
        let mut support_set = SupportSet {
            active_intervals: vec![],
        };
        support_set.recompute(posterior)?;
        Ok(support_set)
    }

    pub fn contains(&self, x: T) -> bool {
        self.active_intervals
            .iter()
            .any(|interval| interval.lower() <= x && x <= interval.upper())
    }

    pub fn widest_interval_midpoint(&self) -> Option<T> {
        let two = T::one() + T::one();

        self.active_intervals
            .iter()
            .max_by(|a, b| {
                let wa = a.upper() - a.lower();
                let wb = b.upper() - b.lower();
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|interval| (interval.lower() + interval.upper()) / two)
    }

    /// Recomputes the support from the posterior distribution.
    ///
    /// # Algorithm
    ///
    /// 1. Convert ε threshold into log-space
    /// 2. Scan posterior log-interval masses
    /// 3. Extract contiguous regions above threshold
    /// 4. Convert index regions into domain intervals
    ///
    /// # Properties
    ///
    /// - Deterministic given posterior state
    /// - Idempotent under unchanged posterior
    /// - No mutation of posterior occurs
    ///
    /// # Numerical considerations
    ///
    /// Operates in log-space to prevent underflow in probability mass.
    pub(crate) fn recompute(
        &mut self,
        posterior: &PosteriorDistribution<T>,
    ) -> Result<(), IntervalError<T>> {
        let eps = T::from_f64(1e-12).unwrap();
        let log_eps = eps.ln();

        self.active_intervals.clear();

        let mut i = 0;

        while i < posterior.log_interval_mass.len() {
            if posterior.log_interval_mass[i] > log_eps {
                let start = i;

                while i < posterior.log_interval_mass.len()
                    && posterior.log_interval_mass[i] > log_eps
                {
                    i += 1;
                }

                let end = i - 1;

                self.active_intervals.push(Interval::new(
                    posterior.knots[start],
                    posterior.knots[end + 1],
                )?);
            } else {
                i += 1;
            }
        }
        Ok(())
    }
}
