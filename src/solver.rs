use crate::{InferenceState, PBError, RootOracle, Scaler, SequentialInterval};

use confi::ConfidenceLevel;
use num_traits::{Float, FromPrimitive};
use std::ops::Range;
use trellis_runner::{CancellationGuard, FallibleProcedure, Progress, TrellisFloat, UserState};

pub struct RootFinder<T> {
    // The scaler for the system.
    //
    // The solver operates internally on range [0, 1] to improve convergence. The [`Scaler`]
    // converts values from this range to the true range of function input.
    scaler: Scaler<T>,

    max_sign_evaluations: usize,
    // The confidence level
    confidence_level: ConfidenceLevel<T>,
}

impl<T: Float + FromPrimitive + std::fmt::Debug> RootFinder<T> {
    pub(crate) fn new(
        domain: Range<T>,
        confidence_level: ConfidenceLevel<T>,
        max_sign_evaluations: usize,
    ) -> Result<Self, PBError<T>> {
        let scaler = Scaler::unit_domain_transform(domain.clone())?;

        Ok(Self {
            scaler,
            max_sign_evaluations,
            confidence_level,
        })
    }

    pub(crate) fn scaled_domain(&self) -> &Range<T> {
        self.scaler.scaled_domain()
    }

    /// Builds an ordered list of candidate query points in scaled coordinates.
    ///
    /// The posterior median is the preferred query point because, in the usual
    /// probabilistic bisection update, it approximately splits the current
    /// posterior mass into two equally likely root-location events.
    ///
    /// However, in noisy problems the median can lie very close to the true root.
    /// In that case the objective sign may be impossible to determine reliably
    /// within the allowed sign-evaluation budget. This method therefore adds a
    /// small number of fallback candidates away from the median.
    ///
    /// Candidate order:
    ///
    /// 1. posterior median
    /// 2. interior quartile-like points of the current sequential confidence interval
    /// 3. posterior quartiles
    /// 4. midpoint of the widest active support interval, if the median is outside support
    ///
    /// Boundary points are removed because observations at domain endpoints do
    /// not refine the posterior partition. Near-duplicate points are also
    /// removed to avoid repeated attempts at effectively the same location.
    ///
    /// Returned values are in the scaler's transformed coordinate system,
    /// usually `[0, 1]`.
    fn query_candidates(&self, state: &InferenceState<T>) -> Vec<T> {
        let mut candidates = Vec::new();

        let median = state.posterior().median();
        candidates.push(median);

        let confidence = state.confidence().current;
        let width = confidence.upper() - confidence.lower();

        if width > T::zero() {
            let four = T::from_f64(4.0).unwrap();

            candidates.push(confidence.lower() + width / four);
            candidates.push(confidence.lower() + width / (T::one() + T::one()));
            candidates.push(confidence.lower() + width * T::from_f64(3.0).unwrap() / four);
        }

        candidates.push(state.posterior().quantile(T::from_f64(0.25).unwrap()));
        candidates.push(state.posterior().quantile(T::from_f64(0.75).unwrap()));

        if !state.support().contains(median)
            && let Some(x) = state.support().widest_interval_midpoint()
        {
            candidates.push(x);
        }

        self.deduplicate_query_candidates(candidates)
    }

    /// Removes invalid, boundary, and near-duplicate query points.
    ///
    /// The probabilistic posterior is defined on the scaled domain, but endpoint
    /// observations do not split an interval and therefore cannot refine the
    /// posterior. This helper keeps only strict interior points.
    fn deduplicate_query_candidates(&self, candidates: Vec<T>) -> Vec<T> {
        let domain = self.scaler.scaled_domain().clone();
        let eps = T::epsilon() * T::from_f64(128.0).unwrap();

        let mut unique = Vec::new();

        'candidate_loop: for x in candidates {
            if x <= domain.start || x >= domain.end {
                continue;
            }

            for y in &unique {
                if (x - *y).abs() <= eps {
                    continue 'candidate_loop;
                }
            }

            unique.push(x);
        }

        unique
    }
}

impl<T> UserState for InferenceState<T>
where
    T: TrellisFloat + Float,
{
    type Float = T;

    fn is_initialised(&self) -> bool {
        self.slope_sign().is_some()
    }

    fn progress(&self) -> Progress<Self::Float> {
        if self.sign_indeterminate() | self.sequential_stalled() {
            Progress::Complete
        } else {
            Progress::Measure(self.width())
        }
    }
}

impl<T, P> FallibleProcedure<P> for RootFinder<T>
where
    T: TrellisFloat
        + Float
        + FromPrimitive
        + std::ops::AddAssign
        + std::iter::Sum
        + Send
        + Sync
        + 'static,
    P: RootOracle<T>,
{
    type Output = SequentialInterval<T>;
    type State = InferenceState<T>;
    type Error = PBError<T>;

    const NAME: &'static str = "Probabilistic bisection";

    fn initialise_fallible(
        &self,
        problem: &mut P,
        state: &mut Self::State,
    ) -> Result<(), Self::Error> {
        let raw_domain = self.scaler.raw_domain();

        let slope = problem
            .slope_sign(raw_domain, self.confidence_level, self.max_sign_evaluations)?
            .ok_or(PBError::IndeterminateSlope {
                x: (raw_domain.start + raw_domain.end) / (T::one() + T::one()),
            })?;

        state.set_slope_sign(slope);

        Ok(())
    }

    fn step_fallible(
        &self,
        problem: &mut P,
        state: &mut Self::State,
        _guard: CancellationGuard<'_>,
    ) -> Result<(), Self::Error> {
        let slope_sign = state.slope_sign().unwrap();

        for scaled in self.query_candidates(state) {
            let raw = self.scaler.to_raw(scaled)?;

            tracing::info!("trying query: scaled={:?}, raw={:?}", scaled, raw);

            let objective_sign =
                match problem.objective_sign(raw, self.confidence_level, self.max_sign_evaluations)
                {
                    Ok(Some(sign)) => sign,

                    Ok(None) | Err(crate::RootError::MaxIterExceeded(_)) => {
                        tracing::debug!(
                            "sign indeterminate at scaled={:?}, raw={:?}; trying fallback",
                            scaled,
                            raw
                        );
                        continue;
                    }

                    Err(e) => return Err(PBError::Oracle(e)),
                };

            let root_side = problem.root_side(objective_sign, slope_sign);

            tracing::info!(
                "accepted query: root_side={:?}, objective_sign={:?}, slope_sign={:?}, raw={:?}",
                root_side,
                objective_sign,
                slope_sign,
                raw
            );

            state.observe(scaled, root_side, self.confidence_level)?;
            return Ok(());
        }

        state.sign_is_indeterminate();
        Ok(())
    }

    fn finalise_fallible(
        &self,
        _problem: &mut P,
        state: &Self::State,
    ) -> Result<Self::Output, Self::Error> {
        let confidence = state.confidence().clone();

        let raw_lower = self.scaler.to_raw(confidence.lower())?;
        let raw_upper = self.scaler.to_raw(confidence.upper())?;

        Ok(SequentialInterval::instantiate(raw_lower..raw_upper))
    }
}
