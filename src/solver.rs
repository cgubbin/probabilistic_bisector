use crate::{InferenceState, ObjectiveSign, PBError, RootOracle, Scaler, SequentialInterval};

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
            .slope_sign(
                &raw_domain,
                self.confidence_level,
                self.max_sign_evaluations,
            )?
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
        // Generate the next sample point at the median of the current posterior distribution and
        // try to calculate the sign of the sample.
        let median = state.posterior().median();

        let scaled = if state.support().contains(median) {
            median
        } else {
            state.support().widest_interval_midpoint().unwrap_or(median)
        };

        // Convert from the domain of the posterior distribution to the domain of the RootOracle
        let raw = self.scaler.to_raw(scaled)?;
        tracing::info!("median: {:?}, raw: {:?}", median, raw);

        let objective_sign =
            match problem.objective_sign(raw, self.confidence_level, self.max_sign_evaluations) {
                Ok(Some(sign)) => sign,
                Ok(None) | Err(crate::RootError::MaxIterExceeded(_)) => {
                    state.sign_indeterminate();
                    println!("sign indeterminate");
                    return Ok(());
                }
                Err(e) => return Err(PBError::Oracle(e)),
            };

        let slope_sign = state.slope_sign().unwrap();
        let root_side = problem.root_side(objective_sign, slope_sign);
        tracing::info!("root side {root_side:?} ({objective_sign:?} {slope_sign:?}) ({raw:?})");

        state.observe(scaled, root_side, self.confidence_level)?;

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

// impl<T, P> FallibleProcedure<P> for RootFinder<T>
// where
//     T: TrellisFloat,
//     P: RootOracle<T>,
// {
//     type Output = SequentialInterval<T>;
//     type State = RootFinderState<T>;
//     type Error = PBError<T>;

//     const NAME: &'static str = "Probabilistic bisection";

//     fn initialise_fallible(
//         &self,
//         _problem: &mut P,
//         _state: &mut Self::State,
//     ) -> Result<(), Self::Error> {
//         todo!()
//     }

//     fn step_fallible(
//         &self,
//         problem: &mut P,
//         state: &mut Self::State,
//         guard: CancellationGuard<'_>,
//     ) -> Result<(), Self::Error> {
//         todo!()
//     }

//     fn finalise_fallible(
//         &self,
//         problem: &mut P,
//         state: &Self::State,
//     ) -> Result<Self::Output, Self::Error> {
//         todo!()
//     }
// }
