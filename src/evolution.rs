use crate::{
    Interval, MeetSemiLattice, PosteriorDistribution, PosteriorError, SemiMeetError,
    SequentialInterval, Sign, SupportSet,
};

use confi::ConfidenceLevel;
use num_traits::{Float, FromPrimitive};

#[derive(thiserror::Error, Debug)]
pub(crate) enum BisectionError<T> {
    #[error("Semi meet error: {0}")]
    SemiMeet(#[from] SemiMeetError<T>),
    #[error("Posterior error: {0}")]
    Posterior(#[from] PosteriorError<T>),
    #[error("Empty hull...")]
    EmptyHull,
}

#[derive(Clone, Debug)]
pub struct InferenceState<T> {
    pub n: usize,
    pub max_n: usize,
    pub posterior: PosteriorDistribution<T>,
    pub support: SupportSet<T>,
    pub confidence: SequentialInterval<T>,
}

impl<T> InferenceState<T>
where
    T: Float + FromPrimitive + std::iter::Sum + std::ops::AddAssign,
{
    /// Advances the inference state by incorporating a new observation.
    ///
    /// This is the single state transition operator of the system:
    ///
    /// ```text
    /// stateₙ → stateₙ₊₁
    /// ```
    ///
    /// It updates three coupled structures:
    ///
    /// 1. Posterior (measure lattice update)
    /// 2. Support (threshold projection)
    /// 3. Confidence (interval meet-semilattice)
    ///
    /// Each component is updated independently but remains
    /// structurally consistent through shared lattice semantics.
    pub fn observe(
        &mut self,
        x: T,
        sign: Sign,
        conf: ConfidenceLevel<T>,
    ) -> Result<(), BisectionError<T>> {
        // -----------------------------
        // 1. Posterior update (measure evolution)
        // -----------------------------
        self.posterior.observe(x, sign, conf)?;

        // -----------------------------
        // 2. Support projection (filter lattice)
        // -----------------------------
        self.support.recompute(&self.posterior);

        // -----------------------------
        // 3. Confidence update (meet-semilattice contraction)
        // -----------------------------
        let candidate = compute_snapshot(&self.posterior, conf, self.n)?;

        let candidate_interval = SequentialInterval { current: candidate };

        self.confidence = self.confidence.clone().meet(candidate_interval)?;

        // -----------------------------
        // 4. Advance time index
        // -----------------------------
        self.n += 1;

        Ok(())
    }
}

pub fn compute_snapshot<T>(
    posterior: &PosteriorDistribution<T>,
    confidence: ConfidenceLevel<T>,
    n: usize,
) -> Result<Interval<T>, BisectionError<T>>
where
    T: Float + FromPrimitive,
{
    // ----------------------------
    // Waeber constants
    // ----------------------------
    let alpha = confidence.significance().into_inner();
    let n1 = T::from_usize(n + 1).unwrap();

    let c = confidence.into_inner();
    let one_minus_c = T::one() - c;
    let two = T::one() + T::one();

    let d = c * (two * c).ln() + one_minus_c * (two * one_minus_c).ln();

    let beta = (c / one_minus_c).ln();

    let b = n1 * d - n1.sqrt() * (-(T::one() / two) * (alpha / two).ln()).sqrt() * beta;

    // ----------------------------
    // numerical stabilisation
    // ----------------------------
    let max_log = posterior
        .log_interval_mass
        .iter()
        .cloned()
        .fold(T::neg_infinity(), T::max);

    let b_shifted = b - max_log;

    // ----------------------------
    // G_n construction (lattice support set)
    // ----------------------------
    let mut g: Vec<usize> = posterior
        .log_interval_mass
        .iter()
        .enumerate()
        .filter(|(_, lp)| **lp - max_log > b_shifted)
        .map(|(i, _)| i)
        .collect();

    let start = *g.first().ok_or(BisectionError::EmptyHull)?;
    let end = *g.last().ok_or(BisectionError::EmptyHull)?;

    let candidate = Interval {
        lower: posterior.knots[start],
        upper: posterior.knots[end + 1],
    };

    Ok(candidate)
}
