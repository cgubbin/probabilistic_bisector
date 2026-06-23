// In the PBA we cannot be certain of the true location of the root. Instead we track a
// probability distribution which represents the sum of our knowledge about it's best location.
//
// The `distribution` module contains an implementation of this distribution. The class is initialised
// with a range of values which can be expected to contain the root with a probability approaching
// unity. It stores the range limits, or samples, and the logarithm of the probability in the
// intermediate region ln(1) = 0. As the algorithm proceeds, new sample points are inserted and
// the probabilities are updated.
mod distribution;
mod error;
mod evolution;
mod root;
mod scaling;
pub(crate) mod semimeet;
mod solver;
pub(crate) mod support;

use evolution::InferenceState;

pub use confi::ConfidenceLevel;
pub use root::{ObjectiveSign, RootError, RootOracle, RootSide};

use confi::SignificanceLevel;
use distribution::{PosteriorDistribution, PosteriorError, PosteriorValidationError};
pub use error::PBError;
pub(crate) use evolution::BisectionError;

use scaling::{Scaler, ScalerError};
use semimeet::{Interval, MeetSemiLattice, SemiMeetError, SequentialInterval};
use support::SupportSet;

use solver::RootFinder;

use num_traits::{Float, FromPrimitive};
use std::ops::Range;
use trellis_runner::{
    EngineOutput, GenerateBuilderFallible, MaxIterationPolicy, RelativeTolerancePolicy,
    TrellisFloat,
};

pub struct BisectorConfig<T> {
    pub max_observations: usize,
    pub max_knots: usize,
    pub max_sign_evaluations: usize,
    pub rel_tol: T,
    pub tolerance_window: usize,
}

pub fn run<T, P>(
    domain: Range<T>,
    confidence_level: ConfidenceLevel<T>,
    problem: P,
    config: BisectorConfig<T>,
) -> Result<EngineOutput<SequentialInterval<T>, InferenceState<T>>, PBError<T>>
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
    let root_finder = RootFinder::new(
        domain.clone(),
        confidence_level,
        config.max_sign_evaluations,
    )?;
    let state = InferenceState::new(root_finder.scaled_domain().clone(), config.max_knots)?;

    let tolerance_window = 10;
    let engine = <RootFinder<T> as GenerateBuilderFallible>::build_for(root_finder, problem)
        .and_policy(MaxIterationPolicy::new(config.max_observations))
        .and_policy(RelativeTolerancePolicy::new(
            config.rel_tol,
            config.tolerance_window,
        ))
        .with_initial_state(state)
        .finalise();

    let result = engine.run();

    match result {
        Ok(output) => Ok(output),
        Err(e) => Err(PBError::Wrapped(Box::new(e))),
    }
}
