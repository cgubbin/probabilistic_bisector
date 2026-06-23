//! # Scaling
//!
//! This module defines the coordinate transform used by the probabilistic
//! bisection algorithm.
//!
//! Internally, the posterior distribution is represented on a fixed,
//! numerically convenient coordinate domain, usually `[0, 1]`. User-provided
//! objective functions, however, are evaluated on their original raw domain.
//!
//! The [`Scaler`] maps between these two coordinate systems:
//!
//! ```text
//! posterior coordinate  <->  raw objective coordinate
//! ```
//!
//! This transformation is not part of the probabilistic model. It is a
//! numerical convenience layer used to keep the posterior representation stable
//! while still evaluating the user’s objective function in its native units.
//!
//! ## Scaling modes
//!
//! Two transforms are supported:
//!
//! - [`ScaleMode::Linear`]: affine scaling between the scaled and raw domains
//! - [`ScaleMode::Log10`]: logarithmic scaling for strictly positive domains
//!   spanning several orders of magnitude
//!
//! Logarithmic scaling is useful when a positive search domain has large
//! multiplicative scale variation, for example `1e-9..1e3`. In such cases,
//! linear scaling would allocate most posterior resolution to large raw values.
//!
//! Negative and sign-changing domains are always scaled linearly.
//!
//! ## Invariants
//!
//! A valid scaler requires:
//!
//! - finite raw bounds
//! - finite scaled bounds
//! - non-empty raw domain
//! - non-empty scaled domain
//! - logarithmic scaling only on strictly positive raw domains
use num_traits::{Float, FromPrimitive};
use std::ops::Range;

#[derive(thiserror::Error, Debug)]
pub(crate) enum ScalerError<T> {
    #[error("invalid domain used to construct scalar: {0:?}")]
    InvalidDomain(Range<T>),
    #[error("invalid value: {value:?}, not in {range:?}")]
    InvalidValue { value: T, range: Range<T> },
}

/// Scaling mode used to map between posterior and raw coordinates.
///
/// The mode determines how interpolation is performed in the raw domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    /// Affine scaling between scaled and raw coordinates.
    Linear,

    /// Base-10 logarithmic scaling.
    ///
    /// This is only valid for strictly positive raw domains.
    Log10,
}

/// Maps between posterior coordinates and raw objective coordinates.
///
/// The posterior distribution is represented on a fixed scaled domain,
/// usually `[0, 1]`, while the user-provided objective function is evaluated on
/// its original raw domain.
///
/// A `Scaler` provides both directions:
///
/// - [`Scaler::to_raw`] maps posterior coordinates to objective coordinates
/// - [`Scaler::to_scaled`] maps objective coordinates back to posterior coordinates
///
/// The transform is bijective over the configured domains.
///
/// # Interpretation
///
/// This type is not part of the inference model. It does not change posterior
/// probabilities, confidence levels, or observation signs. It only changes the
/// coordinate system in which query points are represented.
#[derive(Debug, Clone)]
pub struct Scaler<T> {
    /// Raw objective-function domain.
    raw: Range<T>,

    /// Internal posterior-coordinate domain, usually `0..1`.
    scaled: Range<T>,

    /// Coordinate transform used between the two domains.
    mode: ScaleMode,
}

fn lerp<T: Float>(range: &Range<T>, u: T) -> T {
    range.start + u * (range.end - range.start)
}

fn inv_lerp<T: Float>(range: &Range<T>, x: T) -> T {
    (x - range.start) / (range.end - range.start)
}

impl<T: Float + FromPrimitive> Scaler<T> {
    pub(crate) fn unit_domain_transform(raw: Range<T>) -> Result<Self, ScalerError<T>> {
        Self::new(raw, T::zero()..T::one())
    }

    pub(crate) fn new(raw: Range<T>, scaled: Range<T>) -> Result<Self, ScalerError<T>> {
        if !raw.start.is_finite() || !raw.end.is_finite() || raw.start >= raw.end {
            return Err(ScalerError::InvalidDomain(raw));
        }

        if !scaled.start.is_finite() || !scaled.end.is_finite() || scaled.start >= scaled.end {
            return Err(ScalerError::InvalidDomain(scaled));
        }

        let mode = if raw.start > T::zero() && raw.end / raw.start > T::from_f64(1e3).unwrap() {
            ScaleMode::Log10
        } else {
            ScaleMode::Linear
        };

        Ok(Self { raw, scaled, mode })
    }

    fn mode(&self) -> ScaleMode {
        self.mode
    }

    pub(crate) fn raw_domain(&self) -> &Range<T> {
        &self.raw
    }

    pub(crate) fn scaled_domain(&self) -> &Range<T> {
        &self.scaled
    }

    pub(crate) fn to_raw(&self, scaled: T) -> Result<T, ScalerError<T>>
    where
        T: Float + FromPrimitive,
    {
        if scaled < self.scaled.start || scaled > self.scaled.end {
            return Err(ScalerError::InvalidValue {
                value: scaled,
                range: self.scaled.clone(),
            });
        }

        let u = inv_lerp(&self.scaled, scaled);

        let y = match self.mode {
            ScaleMode::Log10 => {
                let lo = self.raw.start.log10();
                let hi = self.raw.end.log10();
                let y = lerp(&(lo..hi), u);

                T::from_f64(10.0).unwrap().powf(y)
            }
            ScaleMode::Linear => lerp(&self.raw, u),
        };

        Ok(y)
    }

    pub(crate) fn to_scaled(&self, raw: T) -> Result<T, ScalerError<T>>
    where
        T: Float + FromPrimitive,
    {
        if raw < self.raw.start || raw > self.raw.end {
            return Err(ScalerError::InvalidValue {
                value: raw,
                range: self.raw.clone(),
            });
        }

        let u = match self.mode {
            ScaleMode::Linear => inv_lerp(&self.raw, raw),
            ScaleMode::Log10 => {
                let lo = self.raw.start.log10();
                let hi = self.raw.end.log10();

                inv_lerp(&(lo..hi), raw.log10())
            }
        };
        Ok(lerp(&self.scaled, u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn linear_scaler_maps_endpoints() {
        let scaler = Scaler::new(10.0..20.0, 0.0..1.0).unwrap();

        assert_eq!(scaler.mode(), ScaleMode::Linear);
        assert_relative_eq!(scaler.to_raw(0.0).unwrap(), 10.0);
        assert_relative_eq!(scaler.to_raw(1.0).unwrap(), 20.0);
    }

    #[test]
    fn linear_scaler_maps_midpoint() {
        let scaler = Scaler::new(10.0..20.0, 0.0..1.0).unwrap();

        assert_relative_eq!(scaler.to_raw(0.5).unwrap(), 15.0);
        assert_relative_eq!(scaler.to_scaled(15.0).unwrap(), 0.5);
    }

    #[test]
    fn linear_scaler_supports_negative_domains() {
        let scaler = Scaler::new(-20.0..-10.0, 0.0..1.0).unwrap();

        assert_eq!(scaler.mode(), ScaleMode::Linear);
        assert_relative_eq!(scaler.to_raw(0.0).unwrap(), -20.0);
        assert_relative_eq!(scaler.to_raw(1.0).unwrap(), -10.0);
        assert_relative_eq!(scaler.to_raw(0.5).unwrap(), -15.0);
    }

    #[test]
    fn linear_scaler_supports_sign_changing_domains() {
        let scaler = Scaler::new(-1.0..1.0, 0.0..1.0).unwrap();

        assert_eq!(scaler.mode(), ScaleMode::Linear);
        assert_relative_eq!(scaler.to_raw(0.0).unwrap(), -1.0);
        assert_relative_eq!(scaler.to_raw(0.5).unwrap(), 0.0);
        assert_relative_eq!(scaler.to_raw(1.0).unwrap(), 1.0);
    }

    #[test]
    fn positive_narrow_domain_uses_linear_scaling() {
        let scaler = Scaler::new(1.0..100.0, 0.0..1.0).unwrap();

        assert_eq!(scaler.mode(), ScaleMode::Linear);
    }

    #[test]
    fn positive_wide_domain_uses_log_scaling() {
        let scaler = Scaler::new(1e-6..1e3, 0.0..1.0).unwrap();

        assert_eq!(scaler.mode(), ScaleMode::Log10);
    }

    #[test]
    fn log_scaler_maps_endpoints() {
        let scaler = Scaler::new(1e-6..1e3, 0.0..1.0).unwrap();

        assert_relative_eq!(scaler.to_raw(0.0).unwrap(), 1e-6);
        assert_relative_eq!(scaler.to_raw(1.0).unwrap(), 1e3);
    }

    #[test]
    fn log_scaler_maps_midpoint_geometrically() {
        let scaler = Scaler::new(1.0..1e4, 0.0..1.0).unwrap();

        // midpoint in log10 space between 10^0 and 10^4 is 10^2
        assert_relative_eq!(scaler.to_raw(0.5).unwrap(), 100.0);
        assert_relative_eq!(scaler.to_scaled(100.0).unwrap(), 0.5);
    }

    #[test]
    fn to_raw_rejects_values_below_transformed_domain() {
        let scaler = Scaler::new(0.0..10.0, 0.0..1.0).unwrap();

        assert!(matches!(
            scaler.to_raw(-0.1),
            Err(ScalerError::InvalidValue { .. })
        ));
    }

    #[test]
    fn to_raw_rejects_values_above_transformed_domain() {
        let scaler = Scaler::new(0.0..10.0, 0.0..1.0).unwrap();

        assert!(matches!(
            scaler.to_raw(1.1),
            Err(ScalerError::InvalidValue { .. })
        ));
    }

    #[test]
    fn to_scaled_rejects_values_below_raw_domain() {
        let scaler = Scaler::new(0.0..10.0, 0.0..1.0).unwrap();

        assert!(matches!(
            scaler.to_scaled(-1.0),
            Err(ScalerError::InvalidValue { .. })
        ));
    }

    #[test]
    fn to_scaled_rejects_values_above_raw_domain() {
        let scaler = Scaler::new(0.0..10.0, 0.0..1.0).unwrap();

        assert!(matches!(
            scaler.to_scaled(11.0),
            Err(ScalerError::InvalidValue { .. })
        ));
    }

    #[test]
    fn rejects_empty_raw_domain() {
        assert!(matches!(
            Scaler::new(1.0..1.0, 0.0..1.0),
            Err(ScalerError::InvalidDomain(_))
        ));
    }

    #[test]
    fn rejects_reversed_raw_domain() {
        assert!(matches!(
            Scaler::new(2.0..1.0, 0.0..1.0),
            Err(ScalerError::InvalidDomain(_))
        ));
    }

    #[test]
    fn rejects_empty_transformed_domain() {
        assert!(matches!(
            Scaler::new(0.0..1.0, 1.0..1.0),
            Err(ScalerError::InvalidDomain(_))
        ));
    }

    #[test]
    fn rejects_reversed_transformed_domain() {
        assert!(matches!(
            Scaler::new(0.0..1.0, 1.0..0.0),
            Err(ScalerError::InvalidDomain(_))
        ));
    }

    #[test]
    fn rejects_nan_raw_domain() {
        assert!(matches!(
            Scaler::new(f64::NAN..1.0, 0.0..1.0),
            Err(ScalerError::InvalidDomain(_))
        ));
    }

    #[test]
    fn rejects_infinite_raw_domain() {
        assert!(matches!(
            Scaler::new(0.0..f64::INFINITY, 0.0..1.0),
            Err(ScalerError::InvalidDomain(_))
        ));
    }

    #[test]
    fn linear_roundtrip_scaled_to_raw_to_scaled() {
        let scaler = Scaler::new(-10.0..30.0, 0.0..1.0).unwrap();

        for x in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0] {
            let raw = scaler.to_raw(x).unwrap();
            let scaled = scaler.to_scaled(raw).unwrap();

            assert_relative_eq!(scaled, x);
        }
    }

    #[test]
    fn linear_roundtrip_raw_to_scaled_to_raw() {
        let scaler = Scaler::new(-10.0..30.0, 0.0..1.0).unwrap();

        for raw in [-10.0, -5.0, 0.0, 10.0, 25.0, 30.0] {
            let scaled = scaler.to_scaled(raw).unwrap();
            let raw2 = scaler.to_raw(scaled).unwrap();

            assert_relative_eq!(raw2, raw);
        }
    }

    #[test]
    fn log_roundtrip_scaled_to_raw_to_scaled() {
        let scaler = Scaler::new(1e-6..1e6, 0.0..1.0).unwrap();

        for x in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0] {
            let raw = scaler.to_raw(x).unwrap();
            let scaled = scaler.to_scaled(raw).unwrap();

            assert_relative_eq!(scaled, x);
        }
    }

    #[test]
    fn log_roundtrip_raw_to_scaled_to_raw() {
        let scaler = Scaler::new(1e-6..1e6, 0.0..1.0).unwrap();

        for raw in [1e-6, 1e-4, 1e-2, 1.0, 1e3, 1e6] {
            let scaled = scaler.to_scaled(raw).unwrap();
            let raw2 = scaler.to_raw(scaled).unwrap();

            let relative_error = ((raw2 - raw) / raw).abs();
            assert!(
                relative_error < 1e-12,
                "expected {raw2} ≈ {raw}, rel err {relative_error}"
            );
        }
    }

    #[test]
    fn transformed_domain_need_not_be_unit_interval() {
        let scaler = Scaler::new(0.0..10.0, -1.0..1.0).unwrap();

        assert_relative_eq!(scaler.to_raw(-1.0).unwrap(), 0.0);
        assert_relative_eq!(scaler.to_raw(0.0).unwrap(), 5.0);
        assert_relative_eq!(scaler.to_raw(1.0).unwrap(), 10.0);

        assert_relative_eq!(scaler.to_scaled(0.0).unwrap(), -1.0);
        assert_relative_eq!(scaler.to_scaled(5.0).unwrap(), 0.0);
        assert_relative_eq!(scaler.to_scaled(10.0).unwrap(), 1.0);
    }
}
