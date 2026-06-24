use crate::{BisectionError, IntervalError, PosteriorError, RootError, ScalerError};

#[derive(thiserror::Error, Debug)]
pub enum PBError<T> {
    #[error("error in scaler: {0}")]
    Scaler(#[from] ScalerError<T>),
    #[error("error in distribution computation: {0}")]
    Posterior(#[from] PosteriorError<T>),
    #[error("failed to determine the function sign at {x} in less than  iterations")]
    IndeterminateSign { x: T },
    #[error("failed to determine the function slope at {x} in less than  iterations")]
    IndeterminateSlope { x: T },
    #[error("error in bisection: {0}")]
    Bisection(#[from] BisectionError<T>),

    #[error("error in interval: {0}")]
    Interval(#[from] IntervalError<T>),
    #[error("error in oracle: {0}")]
    Oracle(#[from] RootError),
    #[error("in computing the slope of the function, no root was detected in the domain")]
    NoRootDetected,

    #[error(
        "any other variant, wrapped in a box: this is because trellis only returns boxed errors...: {0:?}"
    )]
    Wrapped(Box<dyn std::error::Error + 'static + Send + Sync>),
}
