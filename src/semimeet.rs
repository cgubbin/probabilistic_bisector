//! # Semimeet Module
//!
//! This module defines the algebraic structure used to represent
//! *monotone refinement under uncertainty*.
//!
//! The central abstraction is the **meet-semilattice**, which captures
//! operations where information accumulates and uncertainty shrinks.
//!
//! ## Core idea
//!
//! Elements in this system evolve only by intersection:
//!
//! ```text
//! xₙ₊₁ = xₙ ∧ yₙ
//! ```
//!
//! where `∧` is a *meet* (greatest lower bound).
//!
//! This ensures:
//! - monotonic contraction of intervals
//! - no expansion under updates
//! - stability under repeated inference
//!
//! ## Mathematical structure
//!
//! Each type implementing `MeetSemiLattice` forms a partially ordered set:
//!
//! ```text
//! (L, ≤, ∧)
//! ```
//!
//! where:
//! - `a ∧ b` = greatest lower bound
//! - `a ≤ b` iff a = a ∧ b
//!
//! This guarantees:
//! - idempotence: a ∧ a = a
//! - commutativity: a ∧ b = b ∧ a
//! - associativity
//!
//! ## Design intent
//!
//! This abstraction replaces ad-hoc “clamping”, “validation”, and
//! “projection” logic with a single compositional operation:
//!
//! > all refinement is intersection in a lattice
//!
//! This ensures invariants are enforced structurally, not procedurally.
use num_traits::Float;
///
/// A meet-semilattice defines a structure where elements can only
/// be refined by intersection.
///
/// This is the algebraic backbone of sequential inference:
/// each update can only reduce uncertainty.
pub(crate) trait MeetSemiLattice: Sized {
    /// Error produced when two elements have no valid intersection.
    type Error;

    /// Compute the greatest lower bound (intersection) of two elements.
    ///
    /// # Semantics
    ///
    /// - Returns the intersection of two information states
    /// - Fails if the resulting state is empty or invalid
    ///
    /// # Interpretation
    ///
    /// This represents *information refinement*, not averaging or blending.
    fn meet(self, other: Self) -> Result<Self, Self::Error>;
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum SemiMeetError<T> {
    MeetFailure {
        left: Interval<T>,
        right: Interval<T>,
    },
}

/// A closed interval representing uncertainty over a scalar domain.
///
/// Intervals form a meet-semilattice under intersection.
///
/// # Lattice operation
///
/// ```text
/// [a, b] ∧ [c, d] = [max(a,c), min(b,d)]
/// ```
///
/// # Failure mode
///
/// The meet operation fails if the intervals are disjoint,
/// producing an empty set.
#[derive(Debug, Clone, Copy)]
pub struct Interval<T> {
    pub lower: T,
    pub upper: T,
}

impl<T> MeetSemiLattice for Interval<T>
where
    T: Float,
{
    type Error = SemiMeetError<T>;

    fn meet(self, other: Self) -> Result<Self, Self::Error> {
        let l = self.lower.max(other.lower);
        let r = self.upper.min(other.upper);

        if l > r {
            return Err(SemiMeetError::MeetFailure {
                left: self,
                right: other,
            });
        }

        Ok(Interval { lower: l, upper: r })
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

impl<T> MeetSemiLattice for SequentialInterval<T>
where
    T: Float,
{
    type Error = SemiMeetError<T>;

    fn meet(self, other: Self) -> Result<Self, Self::Error> {
        let c = self.current.meet(other.current)?;

        Ok(Self { current: c })
    }
}
