mod dist;
mod update;

use num_traits::Float;

#[derive(Debug, thiserror::Error)]
pub enum PosteriorError<T> {
    #[error("posterior distributions must contain at least 2 knots.")]
    TooFewknots,
    #[error("posterior distributions can only contain {max_knots} knots")]
    TooManyKnots { max_knots: usize },
    #[error("the provided domain must have finite range.")]
    InvalidDomain,
    #[error("trying to insert a sample point which is too close to points already in the domain.")]
    InsertionPointTooClose,
    #[error("the provided point {0:?} is outside the domain.")]
    PointOutsideDomain(T),
    #[error("invalid state: {0}")]
    ValidationError(#[from] PosteriorValidationError<T>),
    #[error("invalid interval index: {0}")]
    InvalidIntervalIndex(usize),
    #[error("invalid knot location: {0:?}")]
    InvalidKnotLocation(T),
    #[error("point {x:?} does not lie strictly inside interval {interval} [{left:?}, {right:?}]")]
    PointNotInInterval {
        x: T,
        interval: usize,
        left: T,
        right: T,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum PosteriorValidationError<T> {
    #[error("knots not ordered")]
    KnotsNotSorted,
    #[error("inconsistent log_interval_mass length. expected {expected}, found {actual}")]
    IntervalCountMismatch { expected: usize, actual: usize },
    #[error("non-finite log mass")]
    InvalidLogMass,
    #[error("invalid probability value {0:?}")]
    InvalidProbability(T),
    #[error("invalid simplex. found {sum:?}, expected {expected:?} (tol = {tol:?})")]
    SimplexInvalid { sum: T, expected: T, tol: T },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservationLocation {
    /// x coincides with an existing knot.
    ExistingKnot(usize),

    /// x lies strictly inside interval i:
    ///
    /// 'knots[i] < x < knots[i + 1]'
    ///
    Interior(usize),

    /// x coincides with the left or right boundary of the domain.
    Boundary,
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

impl<T> PosteriorDistribution<T> {
    // Create a new [`Distribution`] on domain, with maximum knot count max_knots
    //
    // Initially a uniform distribution is assumed on the domain. In future this could be updated
    // to reflect prior knowledge about the location of the root.
    //
    // # Errors
    // - If max_knots is less than two
    // - If the domain is not ordered
    pub(super) fn new(lower: T, upper: T, max_knots: usize) -> Result<Self, PosteriorError<T>>
    where
        T: Float,
    {
        if max_knots <= 2 {
            return Err(PosteriorError::TooFewknots);
        }
        if (lower >= upper) || lower.is_nan() || upper.is_nan() {
            return Err(PosteriorError::InvalidDomain);
        }
        let result = Self {
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
        };

        Ok(result)
    }

    pub fn log_interval_density(&self, i: usize) -> T
    where
        T: Float,
    {
        let width = self.knots[i + 1] - self.knots[i];
        self.log_interval_mass[i] - width.ln()
    }

    pub fn max_log_interval_density(&self) -> T
    where
        T: Float,
    {
        (0..self.log_interval_mass.len())
            .map(|i| self.log_interval_density(i))
            .fold(T::neg_infinity(), T::max)
    }

    fn locate(&self, x: T) -> Result<ObservationLocation, PosteriorError<T>>
    where
        T: Float,
    {
        let first = self.knots[0];
        let last = self.knots[self.knots.len() - 1];

        if x < first || x > last {
            return Err(PosteriorError::PointOutsideDomain(x));
        }

        if x == first || x == last {
            return Ok(ObservationLocation::Boundary);
        }

        let idx = self
            .knots
            .iter()
            .position(|k| x <= *k)
            .expect("point already verified to lie inside domain");

        if x == self.knots[idx] {
            Ok(ObservationLocation::ExistingKnot(idx))
        } else {
            Ok(ObservationLocation::Interior(idx - 1))
        }
    }

    /// Validates all structural and probabilistic invariants of the posterior.
    ///
    /// This function is intended for debug builds and testing.
    /// It enforces the posterior validity invariant:
    /// - knot ordering
    /// - interval consistency
    /// - probability simplex constraints (approximate)
    fn validate(&self, tol: T) -> Result<(), PosteriorValidationError<T>>
    where
        T: Float,
    {
        // 1. structural invariants (knots)
        if !self.knots.windows(2).all(|w| w[0] < w[1]) {
            return Err(PosteriorValidationError::KnotsNotSorted);
        }

        if self.log_interval_mass.len() + 1 != self.knots.len() {
            return Err(PosteriorValidationError::IntervalCountMismatch {
                expected: self.knots.len() - 1,
                actual: self.log_interval_mass.len(),
            });
        }

        // 2. reconstruct probability vector from log space
        let mut probs: Vec<T> = Vec::with_capacity(self.log_interval_mass.len());

        for lp in &self.log_interval_mass {
            if !lp.is_finite() {
                return Err(PosteriorValidationError::InvalidLogMass);
            }

            let p = lp.exp();

            if !p.is_finite() || p < T::zero() {
                return Err(PosteriorValidationError::InvalidProbability(p));
            }

            probs.push(p);
        }

        // 3. simplex check
        let sum: T = probs.iter().copied().fold(T::zero(), |a, b| a + b);

        if (sum - T::one()).abs() > tol {
            return Err(PosteriorValidationError::SimplexInvalid {
                sum,
                expected: T::one(),
                tol,
            });
        }

        Ok(())
    }
}
