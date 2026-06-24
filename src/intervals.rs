//! # Intervals
//!
//! This module defines the interval types used to represent uncertainty in the
//! probabilistic bisection algorithm.
//!
//! The solver maintains a confidence region for the root. A raw confidence
//! candidate is computed from the current posterior distribution, and the
//! sequential confidence interval is updated by intersecting this candidate
//! with the previous interval.
//!
//! This gives the update rule:
//!
//! ```text
//! Iₙ₊₁ = Iₙ ∩ Cₙ
//! ```
//!
//! where:
//!
//! - `Iₙ` is the current sequential confidence interval,
//! - `Cₙ` is the candidate interval computed from the posterior.
//!
//! The intersection operation is the meet operation on closed intervals:
//!
//! ```text
//! [a, b] ∧ [c, d] = [max(a, c), min(b, d)]
//! ```
//!
//! If the intervals are disjoint, the meet is empty. In that case the caller can
//! either treat the event as an error or keep the previous sequential interval
//! and continue posterior updates.
use num_traits::Float;
use std::ops::Range;

#[derive(thiserror::Error, Debug)]
#[error("meet failure between {left:?} and {right:?}")]
pub struct MeetError<T> {
    left: Interval<T>,
    right: Interval<T>,
}

#[derive(thiserror::Error, Debug)]
pub enum IntervalError<T> {
    #[error("invalid interval bounds: lower={lower:?}, upper={upper:?}")]
    InvalidBounds { lower: T, upper: T },

    #[error("invalid domain: {0:?}")]
    InvalidDomain(Range<T>),
}

/// A closed interval representing uncertainty over a scalar domain.
///
/// The interval invariant is:
///
/// ```text
/// lower <= upper
/// ```
///
/// Intervals are used both as candidate confidence regions and as the underlying
/// state of a [`SequentialInterval`].
#[derive(Debug, Clone, Copy)]
pub struct Interval<T> {
    lower: T,
    upper: T,
}

impl<T> Interval<T> {
    /// Constructs a closed interval from explicit bounds.
    ///
    /// Returns an error if either bound is non-finite or if `lower > upper`.
    pub fn new(lower: T, upper: T) -> Result<Self, IntervalError<T>>
    where
        T: Float,
    {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(IntervalError::InvalidBounds { lower, upper });
        }

        Ok(Self { lower, upper })
    }

    /// Constructs an interval from a half-open [`Range`].
    ///
    /// The range is interpreted only as a pair of domain bounds; the resulting
    /// interval is closed.
    pub fn from_domain(domain: Range<T>) -> Result<Self, IntervalError<T>>
    where
        T: Float,
    {
        if !domain.start.is_finite() || !domain.end.is_finite() || domain.start >= domain.end {
            return Err(IntervalError::InvalidDomain(domain));
        }

        Ok(Self {
            lower: domain.start,
            upper: domain.end,
        })
    }

    /// Lower endpoint.
    pub fn lower(&self) -> T
    where
        T: Copy,
    {
        self.lower
    }

    /// Upper endpoint.
    pub fn upper(&self) -> T
    where
        T: Copy,
    {
        self.upper
    }

    /// Interval width.
    pub fn width(&self) -> T
    where
        T: Float,
    {
        self.upper - self.lower
    }

    /// Returns the intersection of two intervals.
    ///
    /// This is the meet operation:
    ///
    /// ```text
    /// [a, b] ∧ [c, d] = [max(a,c), min(b,d)]
    /// ```
    pub fn meet(self, other: Self) -> Result<Self, MeetError<T>>
    where
        T: Float,
    {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);

        if lower > upper {
            return Err(MeetError {
                left: self,
                right: other,
            });
        }

        Ok(Self { lower, upper })
    }

    /// Returns true if `x` lies inside the closed interval.
    pub fn contains(&self, x: T) -> bool
    where
        T: Float,
    {
        self.lower <= x && x <= self.upper
    }
}

/// A sequential interval represents a monotone sequence of refined
/// confidence regions.
///
/// Each update enforces:
///
/// ```text
/// Iₙ₊₁ ⊆ Iₙ
/// ```
///
/// This ensures:
/// - shrinking uncertainty over time
/// - stability of inference trajectory
/// - consistency with lattice structure
///
/// It is implemented as repeated meet operations over candidate intervals.
#[derive(Clone, Debug)]
pub struct SequentialInterval<T> {
    pub current: Interval<T>,
}

impl<T> SequentialInterval<T> {
    pub(crate) fn instantiate(domain: Range<T>) -> SequentialInterval<T>
    where
        T: Float,
    {
        if domain.start.is_nan() || domain.end.is_nan() {
            panic!("domain contained NaN");
        }

        if domain.start >= domain.end {
            panic!("ill-defined domain");
        }

        Self {
            current: Interval {
                lower: domain.start,
                upper: domain.end,
            },
        }
    }

    pub(crate) fn width(&self) -> T
    where
        T: Float,
    {
        self.current.upper - self.current.lower
    }

    pub(crate) fn lower(&self) -> T
    where
        T: Copy,
    {
        self.current.lower
    }

    pub(crate) fn upper(&self) -> T
    where
        T: Copy,
    {
        self.current.upper
    }

    pub fn try_meet(self, other: Interval<T>) -> Result<Self, MeetError<T>>
    where
        T: Float,
    {
        let lower = self.current.lower.max(other.lower);
        let upper = self.current.upper.min(other.upper);

        if lower > upper {
            return Err(MeetError {
                left: self.current,
                right: other,
            });
        }

        Ok(Self {
            current: Interval { lower, upper },
        })
    }

    pub fn meet_or_keep(self, other: Interval<T>) -> (Self, bool)
    where
        T: Float,
    {
        match self.clone().try_meet(other) {
            Ok(next) => (next, true),
            Err(_) => (self, false),
        }
    }
}
