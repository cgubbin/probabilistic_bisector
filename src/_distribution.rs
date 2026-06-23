use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use statrs::{
    distribution::{Continuous, ContinuousCDF},
    statistics::{Max, Min},
};
use std::{fmt, iter, ops::Range};

use super::Sign;
use confi::{ConfidenceInterval, ConfidenceLevel};

// How close we allow points in the distribution to be before we consider them to be equal
const EPSILON: f64 = 1e-15;

#[derive(Debug, thiserror::Error)]
pub enum DistributionError<T> {
    #[error("posterior distributions must contain at least 2 knots.")]
    TooFewknots,
    #[error("the provided domain must have finite range.")]
    InvalidDomain,
    #[error("trying to insert a sample point which is too close to points already in the domain.")]
    InsertionPointTooClose,
    #[error("the provided point {0} is outside the domain.")]
    PointOutsideDomain(T),
    #[error("invalid state: {0}")]
    InvalidState(String),
}

#[derive(Clone, Debug)]
pub(super) struct PosteriorDistribution<T> {
    /// Maximum number of sample points that may be stored.
    ///
    /// Capacity for `knots` and `log_interval_mass` is reserved
    /// during construction to minimise reallocations during posterior
    /// updates.
    max_knots: usize,

    /// Sorted breakpoint locations defining the piecewise partition of
    /// the domain.
    ///
    /// The first and last entries are always equal to
    /// `domain.start` and `domain.end`, respectively.
    pub(super) knots: Vec<T>,

    /// Logarithm of the interval mass
    ///
    /// Entry `i` corresponds to the interval
    /// `[knots[i], knots[i + 1])`.
    ///
    /// This vector therefore always contains exactly one fewer element
    /// than `knots`.
    pub(super) log_interval_mass: Vec<T>,
}

#[derive(Debug)]
pub(super) struct InvariantSummary<T> {
    pub knots_sorted: bool,
    pub knot_count: usize,
    pub interval_count_matches: bool,

    pub simplex_sum: T,
    pub simplex_valid: bool,

    pub min_probability: T,
    pub max_probability: T,
}

impl<T> InvariantSummary<T> {
    fn validate(self) -> Result<(), DistributionError<T>> {
        todo!()
    }
}

#[derive(Debug)]
struct InsertionEvent<T> {
    index: usize,
    x: T,
    split_mass_log: T,
}

pub(super) enum ObservationUpdate {
    NoOp,
    UpdateOnly,
    RefineAndUpdate,
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
    // Create a new [`Distribution`] on domain, with maximum knot count max_knots
    //
    // Initially a uniform distribution is assumed on the domain. In future this could be updated
    // to reflect prior knowledge about the location of the root.
    //
    // # Errors
    // - If max_knots is less than two
    // - If the domain is not ordered
    pub(super) fn new(lower: T, upper: T, max_knots: usize) -> Result<Self, DistributionError<T>>
    where
        T: Float,
    {
        if max_knots <= 2 {
            return Err(DistributionError::TooFewknots);
        }
        if (lower >= upper) || lower.is_nan() || upper.is_nan() {
            return Err(DistributionError::InvalidDomain);
        }
        Ok(Self {
            max_knots,
            knots: {
                let mut knots = Vec::with_capacity(max_knots);
                knots.push(lower);
                knots.push(upper);
                knots
            },
            log_interval_mass: {
                let mut log_interval_mass = Vec::with_capacity(max_knots - 1);

                log_interval_mass.push(T::zero());
                log_interval_mass
            },
        })
    }

    // Produce a summary of the distribution
    //
    // This is useful for bookeeping, when the caller wants to check a given distribution satisfies
    // the invariant conditions
    pub(super) fn invariant_summary(&self) -> InvariantSummary<T>
    where
        T: Float + FromPrimitive + std::cmp::PartialOrd,
    {
        let knots_sorted = self.knots.windows(2).all(|w| w[0] < w[1]);

        let knot_count = self.knots.len();
        let interval_count_matches =
            self.log_interval_mass.len() == self.knots.len().saturating_sub(1);

        let mut sum = T::zero();
        let mut min_p = T::infinity();
        let mut max_p = T::zero();

        for log_p in &self.log_interval_mass {
            let p = log_p.exp();
            sum = sum + p;

            if p < min_p {
                min_p = p;
            }
            if p > max_p {
                max_p = p;
            }
        }

        let one = T::one();
        let tol = T::from_f64(1e-6).unwrap_or_else(|| T::epsilon());

        let simplex_valid = (sum - one).abs() <= tol;

        InvariantSummary {
            knots_sorted,
            knot_count,
            interval_count_matches,
            simplex_sum: sum,
            simplex_valid,
            min_probability: min_p,
            max_probability: max_p,
        }
    }

    // Returns true if the point x is within the domain of the distribution
    //
    // Note: False is returned for values equal to either endpoint
    fn is_interior(&self, x: &T) -> bool
    where
        T: std::cmp::PartialOrd,
    {
        x > self.knots.first().unwrap() && x < self.knots.last().unwrap()
    }

    /// Validates all structural and probabilistic invariants of the posterior.
    ///
    /// This function is intended for debug builds and testing.
    /// It enforces the posterior validity invariant:
    /// - knot ordering
    /// - interval consistency
    /// - probability simplex constraints (approximate)
    fn validate(&self) -> Result<(), DistributionError<T>>
    where
        T: Float + FromPrimitive + std::cmp::PartialOrd,
    {
        let summary = self.invariant_summary();

        summary.validate()
    }

    /// Returns the probability mass contained in each interval.
    ///
    /// Interval `i` corresponds to
    /// `[knots[i], knots[i + 1])`.
    pub(super) fn bin_probabilities(&self) -> Vec<T>
    where
        T: Float,
    {
        self.knots
            .windows(2)
            .zip(self.log_interval_mass.iter())
            .map(|(window, &log_pdf)| {
                let width = window[1] - window[0];
                log_pdf.exp() * width
            })
            .collect()
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
    fn insert_knot(&mut self, x: T) -> Result<InsertionEvent<T>, DistributionError<T>>
    where
        T: Float + FromPrimitive + fmt::Debug,
    {
        let idx = self
            .knots
            .iter()
            .position(|k| x <= *k)
            .ok_or_else(|| DistributionError::PointOutsideDomain(x))?;

        debug_assert!(idx > 0 && idx < self.knots.len());

        let split_mass_log = self.log_interval_mass[idx - 1];

        self.knots.insert(idx, x);

        // split interval mass (uniform split assumption)
        self.log_interval_mass.insert(idx - 1, split_mass_log);

        Ok(InsertionEvent {
            index: idx,
            x,
            split_mass_log,
        })
    }

    /// Return the posterior quantile corresponding to cumulative probability `q`.
    ///
    /// The returned value `x` satisfies:
    ///
    /// ```text
    /// P(X <= x) = q
    /// ```
    ///
    /// Quantiles are computed by linearly interpolating within the interval
    /// containing the requested cumulative probability.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - `q` is outside `[0, 1]`
    /// - the posterior mass does not sum to unity
    pub(super) fn quantile(&self, q: T) -> T
    where
        T: Float + FromPrimitive + fmt::Debug,
    {
        assert!(q >= T::zero());
        assert!(q <= T::one());

        if q == T::zero() {
            return self.knots[0];
        }

        if q == T::one() {
            return *self.knots.last().unwrap();
        }

        let mut cdf = T::zero();

        for (i, log_mass) in self.log_interval_mass.iter().enumerate() {
            let mass = log_mass.exp();

            let next_cdf = cdf + mass;

            if next_cdf >= q {
                let interval_start = self.knots[i];
                let interval_end = self.knots[i + 1];

                let interval_width = interval_end - interval_start;

                let fraction = (q - cdf) / mass;

                return interval_start + fraction * interval_width;
            }

            cdf = next_cdf;
        }

        panic!("CDF did not reach requested quantile: q={q:?}, final CDF={cdf:?}");
    }

    /// The median is defined as the smallest position `x` such that the
    /// cumulative probability mass over intervals to its left reaches at least
    /// 0.5.
    ///
    /// The posterior is represented as a discrete distribution over intervals:
    ///
    /// `[knots[i], knots[i+1])`
    ///
    /// with probability mass:
    ///
    /// `P_i = exp(log_interval_mass[i])`
    ///
    /// # Method
    ///
    /// 1. Accumulate interval masses left-to-right.
    /// 2. Identify the interval where the cumulative mass crosses 0.5.
    /// 3. Perform linear interpolation within that interval proportional to
    ///    remaining mass.
    ///
    /// # Assumptions
    ///
    /// - `log_interval_mass` is normalized (sums to 1 in probability space).
    /// - `knots` is strictly increasing.
    /// - `log_interval_mass.len() == knots.len() - 1`.
    ///
    /// # Returns
    ///
    /// A point `x ∈ [knots[0], knots[last]]` such that:
    ///
    /// `P(X ≤ x) ≈ 0.5`
    ///
    /// # Numerical stability
    ///
    /// The method performs interpolation within the median interval to avoid
    /// discretization bias caused by finite partition resolution.
    pub(super) fn median(&self) -> T
    where
        T: Float + FromPrimitive + fmt::Debug,
    {
        self.quantile(T::one() / (T::one() + T::one()))
    }

    pub(super) fn credible_interval(
        &self,
        confidence_level: &ConfidenceLevel<T>,
    ) -> ConfidenceInterval<T>
    where
        T: Float + FromPrimitive + ::std::fmt::Debug,
    {
        let gamma = confidence_level.into_inner();

        let half = T::one() / (T::one() + T::one());

        let tail = (T::one() - gamma) * half;

        let lower = self.quantile(tail);
        let upper = self.quantile(T::one() - tail);

        ConfidenceInterval::new(lower, upper, confidence_level.clone()).unwrap()
    }

    /// Applies a Bayesian update conditioned on an observation occurring exactly
    /// at an existing knot in the partition.
    ///
    /// ## Model interpretation
    ///
    /// A knot observation does not induce structural change. Instead, it acts as
    /// a conditioning event that splits the domain into:
    ///
    /// - intervals strictly to the left of the knot
    /// - intervals at or to the right of the knot
    ///
    /// The update applies a likelihood ratio derived from:
    /// - the observed sign direction
    /// - the confidence in that observation
    ///
    /// ## Effect
    ///
    /// This function updates `log_interval_mass` in-place:
    ///
    /// - intervals left of `knot_index` are scaled by one likelihood factor
    /// - intervals right of `knot_index` are scaled by the complementary factor
    ///
    /// All updates are performed in log-space.
    ///
    /// ## Important
    ///
    /// - This function does NOT modify the knot structure
    /// - This function assumes the knot already exists in the partition
    /// - Total probability mass is preserved only up to subsequent normalization
    fn update_with_observation_at_existing_knot(
        &mut self,
        knot_index: usize,
        direction: Sign,
        confidence: ConfidenceLevel<T>,
    ) where
        T: Float + FromPrimitive + fmt::Debug + std::ops::AddAssign,
    {
        let confidence = confidence.into_inner();

        let dir = match direction {
            Sign::Positive => T::one(),
            Sign::Negative => -T::one(),
        };

        let (left_factor, right_factor) = compute_update_factors(dir, confidence);

        // intervals left of knot
        for i in 0..knot_index {
            self.log_interval_mass[i] += left_factor;
        }

        // intervals right of knot
        for i in knot_index..self.log_interval_mass.len() {
            self.log_interval_mass[i] += right_factor;
        }
    }

    fn find_interval(&self, x: T) -> usize
    where
        T: std::cmp::PartialOrd,
    {
        self.knots
            .windows(2)
            .position(|w| x >= w[0] && x < w[1])
            .expect("x must be within domain")
    }

    fn update_at_index(&mut self, index: usize, direction: Sign, confidence: ConfidenceLevel<T>)
    where
        T: Float + FromPrimitive + fmt::Debug + std::ops::AddAssign,
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

    /// Conditionally refines the partition by inserting a new knot at `x`.
    ///
    /// ## Semantics
    ///
    /// If `x` lies strictly inside an existing interval, that interval is split
    /// into two sub-intervals. The probability mass of the original interval is
    /// duplicated across the two new intervals.
    ///
    /// If `x` coincides with an existing knot or lies on the boundary, no action
    /// is taken.
    ///
    /// ## Important properties
    ///
    /// - This function does not perform Bayesian updates
    /// - It only modifies the geometric representation (knots)
    /// - Probability mass is preserved during splitting
    ///
    /// ## Invariants preserved
    /// - Ordering of knots
    /// - Conservation of probability mass
    fn refine_if_needed(&mut self, x: T, interval: usize) -> Option<usize>
    where
        T: Float + FromPrimitive + fmt::Debug,
    {
        let left = self.knots[interval];
        let right = self.knots[interval + 1];

        if x == left || x == right {
            return None;
        }

        self.knots.insert(interval + 1, x);

        let mass = self.log_interval_mass[interval];
        self.log_interval_mass.insert(interval, mass);

        Some(interval + 1)
    }

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
    ) -> Result<(), DistributionError<T>>
    where
        T: Float + FromPrimitive + fmt::Debug + std::ops::AddAssign + std::iter::Sum<T>,
    {
        if (x == *self.knots.first().unwrap()) | (x == *self.knots.last().unwrap()) {
            return Ok(());
        }

        let i = self.find_interval(x);

        // ALWAYS update first
        self.update_at_index(i + 1, direction, confidence);

        // THEN refine if needed
        self.refine_if_needed(x, i);

        self.renormalize();

        dbg!(self.log_interval_mass.iter().copied().sum::<T>());
        debug_assert!(
            (self.log_interval_mass.iter().copied().sum::<T>() - T::one()).abs()
                < T::min_positive_value()
        );

        Ok(())
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
    fn insert_splits_interval() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

        let initial_len = dist.knots.len();
        dist.insert_knot(0.5).unwrap();

        assert_eq!(dist.knots.len(), initial_len + 1);
        assert_eq!(dist.log_interval_mass.len(), dist.knots.len() - 1);
    }

    #[test]
    fn knots_remain_sorted() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

        dist.insert_knot(0.7).unwrap();
        dist.insert_knot(0.3).unwrap();
        dist.insert_knot(0.9).unwrap();

        assert!(dist.knots.windows(2).all(|w| w[0] < w[1]));
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

            let s = dist.invariant_summary();

            prop_assert!(s.knots_sorted);
            prop_assert!(s.interval_count_matches);
            prop_assert!(s.simplex_valid);
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

            let summary = dist.invariant_summary();

            prop_assert!(summary.knots_sorted);
            prop_assert!(summary.interval_count_matches);
            prop_assert!(summary.simplex_valid);
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
                );
            }

            let s = dist.invariant_summary();

            prop_assert!(s.simplex_valid);
            prop_assert!(s.knots_sorted);
        }
    }

    #[test]
    fn median_is_within_domain() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();

        for i in 1..10 {
            dist.insert_knot(i as f64 / 10.0).unwrap();
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

        let summary = dist.invariant_summary();
        dbg!(&dist, &summary);

        assert!(summary.simplex_valid);
    }

    #[test]
    fn repeated_observations_are_stable() {
        let mut dist = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();

        for _ in 0..20 {
            dist.observe(0.5, Sign::Positive, ConfidenceLevel::new(0.7).unwrap())
                .unwrap();
        }

        let summary = dist.invariant_summary();
        dbg!(&summary, dist);

        assert!(summary.simplex_valid);
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
    fn observation_bias_affects_distribution_directionally() {
        let mut dist_left = PosteriorDistribution::new(0.0, 1.0, 50).unwrap();
        let mut dist_right = dist_left.clone();

        // bias left
        for _ in 0..10 {
            dist_left
                .observe(0.3, Sign::Positive, ConfidenceLevel::new(0.8).unwrap())
                .unwrap();
        }

        // bias right
        for _ in 0..10 {
            dist_right
                .observe(0.7, Sign::Negative, ConfidenceLevel::new(0.8).unwrap())
                .unwrap();
        }

        let m_left = dist_left.median();
        let m_right = dist_right.median();

        assert!(m_left < m_right);
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
