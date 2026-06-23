use crate::{
    Interval, MeetSemiLattice, ObjectiveSign, PBError, PosteriorDistribution, PosteriorError,
    RootSide, SemiMeetError, SequentialInterval, SupportSet,
};

use confi::ConfidenceLevel;
use num_traits::{Float, FromPrimitive};
use std::ops::Range;

#[derive(thiserror::Error, Debug)]
pub enum BisectionError<T> {
    #[error("Semi meet error: {0}")]
    SemiMeet(#[from] SemiMeetError<T>),
    #[error("Posterior error: {0}")]
    Posterior(#[from] PosteriorError<T>),
    #[error("Empty hull...")]
    EmptyHull,
}

#[derive(Clone, Debug)]
pub struct InferenceState<T> {
    iter: usize,
    sign_indeterminate: bool,
    posterior: PosteriorDistribution<T>,
    support: SupportSet<T>,
    confidence: SequentialInterval<T>,
    slope_sign: Option<ObjectiveSign>,
    sequential_stalled: bool,
    empty_meet_count: usize,
}

impl<T: std::fmt::Debug> std::fmt::Display for InferenceState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T> InferenceState<T> {
    pub(crate) fn new(domain: Range<T>, max_knots: usize) -> Result<Self, PBError<T>>
    where
        T: Float + FromPrimitive,
    {
        let posterior = PosteriorDistribution::new(domain.start, domain.end, max_knots)?;
        let support = SupportSet::new(&posterior);
        Ok(Self {
            iter: 0,
            sign_indeterminate: false,
            posterior,
            support,
            confidence: SequentialInterval::instantiate(domain),
            slope_sign: None,
            sequential_stalled: false,
            empty_meet_count: 0,
        })
    }

    pub(crate) fn set_slope_sign(&mut self, sign: ObjectiveSign) {
        self.slope_sign = Some(sign);
    }

    pub(crate) fn slope_sign(&self) -> Option<ObjectiveSign> {
        self.slope_sign
    }

    pub(crate) fn sequential_stalled(&self) -> bool {
        self.sequential_stalled
    }

    pub(crate) fn sign_indeterminate(&self) -> bool {
        self.sign_indeterminate
    }

    pub(crate) fn sign_is_indeterminate(&mut self) {
        self.sign_indeterminate = true;
    }

    pub(crate) fn posterior(&self) -> &PosteriorDistribution<T> {
        &self.posterior
    }

    pub(crate) fn support(&self) -> &SupportSet<T> {
        &self.support
    }

    pub(crate) fn confidence(&self) -> &SequentialInterval<T> {
        &self.confidence
    }

    pub(crate) fn width(&self) -> T
    where
        T: Float,
    {
        self.confidence.width()
    }

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
    pub(crate) fn observe(
        &mut self,
        x: T,
        root_side: RootSide,
        conf: ConfidenceLevel<T>,
    ) -> Result<(), BisectionError<T>>
    where
        T: Float + FromPrimitive + std::iter::Sum + std::ops::AddAssign + std::fmt::Debug,
    {
        // -----------------------------
        // 1. Posterior update (measure evolution)
        // -----------------------------
        tracing::debug!("updating posterior");
        self.posterior.observe(x, root_side, conf)?;

        // -----------------------------
        // 2. Support projection (filter lattice)
        // -----------------------------
        tracing::debug!("recomputing support");
        self.support.recompute(&self.posterior);

        // -----------------------------
        // 3. Confidence update (meet-semilattice contraction)
        // -----------------------------
        let candidate = compute_snapshot(&self.posterior, conf, self.iter)?;

        let candidate_interval = SequentialInterval { current: candidate };

        let (next_confidence, met) = self.confidence.clone().meet_or_keep(candidate);
        self.confidence = next_confidence;

        dbg!(
            &self.iter,
            &self.confidence,
            &candidate,
            candidate.width(),
            self.confidence.width(),
        );

        if !met {
            self.sequential_stalled = true;
            self.empty_meet_count += 1;
        }

        // -----------------------------
        // 4. Advance time index
        // -----------------------------
        self.iter += 1;

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
    let max_log_density = posterior.max_log_interval_density();

    let b_shifted = b - max_log_density;

    // ----------------------------
    // G_n construction (lattice support set)
    // ----------------------------

    let g: Vec<usize> = (0..posterior.log_interval_mass.len())
        .filter(|&i| posterior.log_interval_density(i) - max_log_density > b_shifted)
        .collect();

    dbg!(&g);

    let start = *g.first().ok_or(BisectionError::EmptyHull)?;
    let end = *g.last().ok_or(BisectionError::EmptyHull)?;

    let candidate = Interval {
        lower: posterior.knots[start],
        upper: posterior.knots[end + 1],
    };

    Ok(candidate)
}
