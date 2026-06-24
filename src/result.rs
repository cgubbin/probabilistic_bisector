use trellis_runner::{RunSummary, Termination};

use crate::{InferenceState, Interval, PBError};

#[derive(Clone, Debug)]
pub struct ProbabilisticBisectionResult<T> {
    pub interval: Interval<T>,
    pub termination: Termination,
    pub summary: RunSummary<T>,
    // pub diagnostics: SolverDiagnostics<T>,
}

#[derive(Clone, Debug)]
pub struct SolverDiagnostics<T> {
    pub observations: usize,
    pub knots: usize,
    pub final_scaled_interval: Interval<T>,
    pub final_posterior_median: T,
    pub sign_indeterminate: bool,
}

#[derive(thiserror::Error, Debug)]
pub enum ProbabilisticBisectionError<T> {
    #[error("error in problem preprocessing: {0}")]
    Preprocessing(#[from] PBError<T>),

    #[error("error during solver execution: {error}")]
    Running {
        error: PBError<T>,
        summary: RunSummary<T>,
        state: InferenceState<T>,
    },
}
