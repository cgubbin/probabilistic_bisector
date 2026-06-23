use crate::{PosteriorError, ScalerError};

#[derive(thiserror::Error, Debug)]
pub enum PBError<T> {
    #[error("error in scaler: {0}")]
    Scaler(#[from] ScalerError<T>),
    #[error("error in distribution computation: {0}")]
    Posterior(#[from] PosteriorError<T>),
    #[error("failed to determine the function sign at {0} in less than {1} iterations")]
    SignDetermination(T, usize),
    #[error("in computing the slope of the function, no root was detected in the domain")]
    NoRootDetected,
}
