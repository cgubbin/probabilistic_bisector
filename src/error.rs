use crate::CombinedConfidenceInterval;
use crate::DistributionError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failure as the next sample overlaps with previous ones.\n
    this typically means the window width requested is too small for the problem\n
    and that the requested level of accuracy cannot be achieved.\n
    The result enclosed in the error will enclose the mean, but will not have reached the tolerance requested")]
    OverlappingSamples(CombinedConfidenceInterval<f64>),
    #[error("the convex hull used for the update is empty\n
    this typically means the algorithm is focussed on a window which is too narrow\n
    and the distribution has acquired multiple peaks.\n
    The result enclosed in the error will enclose the mean, but will not have reached the tolerance requested")]
    EmptyHull(CombinedConfidenceInterval<f64>),
    #[error("error in distribution computation: {0}")]
    DistributionError(#[from] DistributionError),
    #[error("failed to determine the function sign at {0} in less than {1} iterations")]
    SignDeterminationError(f64, usize),
    #[error("in computing the slope of the function, no root was detected in the domain")]
    NoRootDetected,
}
