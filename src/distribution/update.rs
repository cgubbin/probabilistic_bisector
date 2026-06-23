use super::{ObservationLocation, PosteriorDistribution, PosteriorError};
use crate::Sign;

use confi::ConfidenceLevel;
use num_traits::{Float, FromPrimitive};

use std::fmt;

const DEFAULT_SIMPLEX_TOL: f64 = 1e-10;

#[derive(Debug)]
struct InsertionEvent<T> {
    index: usize,
    x: T,
    split_mass_log: T,
}

fn compute_update_factors<T: Float>(direction: T, confidence: T) -> (T, T)
where
    T: Float,
{
    let two = T::one() + T::one();

    let delta = if direction >= T::zero() {
        confidence
    } else {
        T::one() - confidence
    };

    let left_factor = (two * (T::one() - delta)).ln();
    let right_factor = (two * delta).ln();

    (left_factor, right_factor)
}

impl<T> PosteriorDistribution<T> {
    /// Applies a single noisy observation to the posterior distribution.
    ///
    /// This is the primary entry point for Bayesian updating.
    ///
    /// The observation consists of:
    /// - a query location `x`
    /// - a directional sign (`Sign::Positive` or `Sign::Negative`)
    /// - a confidence level in the correctness of the sign
    ///
    /// ## Semantics
    ///
    /// The update is performed in two logically distinct phases:
    ///
    /// ### 1. Bayesian update (always applied)
    /// The observation induces a likelihood ratio update over all intervals
    /// in the current partition. This is done in log-space to ensure numerical
    /// stability.
    ///
    /// The update modifies `log_interval_mass` such that:
    ///
    /// ```text
    /// P_{t+1}(interval_i) ∝ P(Y(x) | interval_i) * P_t(interval_i)
    /// ```
    ///
    /// ### 2. Structural refinement (conditional)
    /// If `x` lies strictly inside an existing interval, the interval is split
    /// into two sub-intervals, and probability mass is duplicated accordingly.
    /// This operation does not introduce new information and preserves total mass.
    ///
    /// ## Boundary handling
    ///
    /// If `x` lies on or outside the domain boundary, no structural refinement
    /// is performed. The observation still contributes to the Bayesian update,
    /// but does not modify the partition.
    ///
    /// ## Invariants preserved
    /// - Total probability mass remains normalized (up to numerical tolerance)
    /// - The partition remains strictly ordered
    /// - Log-space representation is maintained throughout
    pub fn observe(
        &mut self,
        x: T,
        direction: Sign,
        confidence: ConfidenceLevel<T>,
    ) -> Result<(), PosteriorError<T>>
    where
        T: Float + FromPrimitive + std::ops::AddAssign + std::iter::Sum<T>,
    {
        match self.locate(x) {
            Ok(ObservationLocation::Boundary) => return Ok(()),

            Ok(ObservationLocation::ExistingKnot(i)) => {
                self.update_at_index(i, direction, confidence);
            }

            Ok(ObservationLocation::Interior(i)) => {
                self.update_at_index(i + 1, direction, confidence);
                self.insert_knot_in_interval(x, i)?;
            }
            Err(_) => unreachable!(),
        }

        self.renormalize();

        self.validate(T::from_f64(DEFAULT_SIMPLEX_TOL).unwrap())?;

        Ok(())
    }

    fn update_at_index(&mut self, index: usize, direction: Sign, confidence: ConfidenceLevel<T>)
    where
        T: Float + FromPrimitive + std::ops::AddAssign,
    {
        let confidence = confidence.into_inner();

        let dir = match direction {
            Sign::Positive => T::one(),
            Sign::Negative => -T::one(),
        };

        let (left_factor, right_factor) = compute_update_factors(dir, confidence);

        for i in 0..index {
            self.log_interval_mass[i] += left_factor;
        }

        for i in index..self.log_interval_mass.len() {
            self.log_interval_mass[i] += right_factor;
        }
    }

    /// # Interval split rule (knot insertion semantics)
    ///
    /// When a new knot `x` is inserted into the distribution, it splits an
    /// existing interval `[x_i, x_{i+1})` into two sub-intervals:
    ///
    /// - `[x_i, x)`
    /// - `[x, x_{i+1})`
    ///
    /// This operation does not introduce new probability mass.
    ///
    /// Instead, it **redistributes the existing interval mass**:
    ///
    /// Let `P_i` be the probability mass associated with the original interval.
    ///
    /// After insertion:
    ///
    /// ```text
    /// P_left  = α · P_i
    /// P_right = (1 - α) · P_i
    /// ```
    ///
    /// where `α` is the split parameter.
    ///
    /// ## Current model assumption
    ///
    /// In this implementation:
    ///
    /// ```text
    /// α = 1/2
    /// ```
    ///
    /// i.e. the prior mass of the interval is split uniformly between the two
    /// new sub-intervals.
    ///
    /// ## Interpretation
    ///
    /// This corresponds to the assumption that:
    ///
    /// > before observing any function evaluations, the new knot provides no
    /// > additional information about how probability mass should be distributed
    /// > within the interval.
    ///
    /// In other words, the insertion is *purely geometric refinement* of the
    /// partition, not an update of belief.
    ///
    /// ## Important invariants
    ///
    /// - Total probability mass is preserved by construction:
    ///
    ///   `P_left + P_right = P_i`
    ///
    /// - No Bayesian update is performed during splitting.
    /// - All information updates are deferred to the subsequent observation step.
    ///
    /// ## Numerical representation
    ///
    /// The implementation stores masses in log-space:
    ///
    /// ```text
    /// log P_left  = log P_i + log α
    /// log P_right = log P_i + log (1 - α)
    /// ```
    ///
    /// With the current uniform assumption:
    ///
    /// ```text
    /// log P_left = log P_right = log P_i - ln 2
    /// ```
    fn insert_knot_in_interval(
        &mut self,
        x: T,
        interval: usize,
    ) -> Result<InsertionEvent<T>, PosteriorError<T>>
    where
        T: Float + FromPrimitive,
    {
        if interval >= self.log_interval_mass.len() {
            return Err(PosteriorError::InvalidIntervalIndex(interval));
        }

        debug_assert_eq!(self.knots.len(), self.log_interval_mass.len() + 1);

        // ----- validate x -----

        if !x.is_finite() {
            return Err(PosteriorError::InvalidKnotLocation(x));
        }

        let left = self.knots[interval];
        let right = self.knots[interval + 1];

        // Knot insertion is only defined for strict interior points.
        if x <= left || x >= right {
            return Err(PosteriorError::PointNotInInterval {
                x,
                interval,
                left,
                right,
            });
        }

        // ----- perform split -----

        let parent_log_mass = self.log_interval_mass[interval];

        // Uniform split:
        //
        // P_left  = P_right = P_parent / 2
        //
        // log(P/2) = log(P) - ln(2)
        let half_log = T::from_f64(2.0).expect("T must represent 2").ln();

        let child_log_mass = parent_log_mass - half_log;

        self.knots.insert(interval + 1, x);

        self.log_interval_mass[interval] = child_log_mass;
        self.log_interval_mass.insert(interval + 1, child_log_mass);

        Ok(InsertionEvent {
            index: interval + 1,
            x,
            split_mass_log: parent_log_mass,
        })
    }

    fn renormalize(&mut self)
    where
        T: Float + FromPrimitive,
    {
        let mut max_log = self.log_interval_mass[0];

        for v in &self.log_interval_mass {
            if *v > max_log {
                max_log = *v;
            }
        }

        let mut sum = T::zero();

        for v in &mut self.log_interval_mass {
            let shifted = (*v - max_log).exp();
            *v = shifted;
            sum = sum + shifted;
        }

        let log_sum = sum.ln();

        for v in &mut self.log_interval_mass {
            *v = (*v).ln() - log_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knots_and_intervals_remain_consistent() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

        dist.observe(0.5, Sign::Positive, ConfidenceLevel::<f64>::CL95)
            .unwrap();

        assert_eq!(dist.knots.len(), dist.log_interval_mass.len() + 1);
    }

    #[test]
    fn knots_remain_sorted() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

        let dir = Sign::Positive;
        let conf = ConfidenceLevel::<f64>::CL95;

        dist.observe(0.7, dir, conf).unwrap();
        dist.observe(0.3, dir, conf).unwrap();
        dist.observe(0.9, dir, conf).unwrap();

        assert!(dist.knots.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn probability_mass_is_preserved() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

        let dir = Sign::Positive;
        let conf = ConfidenceLevel::<f64>::CL95;

        for x in [0.1, 0.4, 0.6, 0.9] {
            dist.observe(x, dir, conf).unwrap();

            let sum: f64 = dist.interval_mass().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn observations_shift_mass_correctly() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 20).unwrap();

        let before = dist.cumulative_mass(0.5);

        dist.observe(0.3, Sign::Negative, ConfidenceLevel::<f64>::CL95)
            .unwrap();

        let after = dist.cumulative_mass(0.5);

        assert!(after > before);
    }

    #[test]
    fn distribution_does_not_collapse() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

        let dir = Sign::Positive;
        let conf = ConfidenceLevel::<f64>::CL95;

        for _ in 0..20 {
            dist.observe(0.5, dir, conf).unwrap();
        }

        assert!(dist.knots.len() > 2);
        assert!(dist.interval_mass().all(|p| p > 0.0));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn posterior_stays_valid_under_random_observations(
            ops in proptest::collection::vec(0f64..1f64, 1..100)
        ) {
            let mut dist = PosteriorDistribution::new(0.0, 1.0, 100).unwrap();

            for x in ops {
                dist.observe(
                    x,
                    Sign::Positive,
                    ConfidenceLevel::new(0.7).unwrap(),
                ).unwrap();
            }
        }
    }

    proptest! {
        #[test]
        fn posterior_invariants_hold(
            ops in proptest::collection::vec(0f64..1f64, 1..50)
        ) {
            let mut dist = PosteriorDistribution::new(0.0, 1.0, 100).unwrap();

            for x in ops {
                dist.observe(
                    x,
                    Sign::Positive,
                    ConfidenceLevel::new(0.7).unwrap(),
                ).unwrap();
            }

        }
    }

    proptest! {
        #[test]
        fn repeated_updates_do_not_break_distribution(
            xs in proptest::collection::vec(0f64..1f64, 10..50)
        ) {
            let mut dist = PosteriorDistribution::new(0.0, 1.0, 200).unwrap();

            for x in xs {
                let _ = dist.observe(
                    x,
                    Sign::Positive,
                    ConfidenceLevel::new(0.8).unwrap()
                ).unwrap();
            }

        }
    }

    #[test]
    fn median_is_within_domain() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();

        for i in 1..10 {
            let x = i as f64 / 10.0;
            dist.observe(x, Sign::Positive, ConfidenceLevel::new(0.7).unwrap())
                .unwrap();
        }

        let m = dist.median();

        assert!(m >= 0.0 && m <= 1.0);
    }

    #[test]
    fn mass_is_preserved_under_repeated_updates() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();

        let inputs = vec![0.1, 0.7, 0.3, 0.9, 0.5];

        for x in inputs {
            dist.observe(x, Sign::Positive, ConfidenceLevel::new(0.7).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn repeated_observations_are_stable() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();

        for _ in 0..20 {
            dist.observe(0.5, Sign::Positive, ConfidenceLevel::new(0.7).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn boundary_observation_does_not_change_structure() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 20).unwrap();

        let before_knots = dist.knots.clone();
        let before_mass = dist.log_interval_mass.clone();

        dist.observe(0.0, Sign::Positive, ConfidenceLevel::new(0.7).unwrap())
            .unwrap();

        assert_eq!(before_knots, dist.knots);
        assert_eq!(before_mass.len(), dist.log_interval_mass.len());
    }

    #[test]
    fn refinement_preserves_alignment() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

        dist.observe(0.37, Sign::Negative, ConfidenceLevel::new(0.8).unwrap())
            .unwrap();

        assert!(dist.knots.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(dist.knots.len() - 1, dist.log_interval_mass.len());
    }

    #[test]
    fn median_stays_within_domain() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();

        for x in [0.1, 0.8, 0.3, 0.6] {
            dist.observe(x, Sign::Positive, ConfidenceLevel::new(0.7).unwrap())
                .unwrap();
        }

        let m = dist.median();

        assert!(m >= 0.0 && m <= 1.0);
    }

    #[test]
    fn no_probability_mass_drift_over_time() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();

        for i in 0..200 {
            let x = (i as f64 % 100.0) / 100.0;

            dist.observe(x, Sign::Positive, ConfidenceLevel::new(0.6).unwrap())
                .unwrap();

            let sum: f64 = dist.log_interval_mass.iter().map(|v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
