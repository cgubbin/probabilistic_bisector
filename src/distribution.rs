// This module describes the probability distribution underlying the probabalistic bisection
// algorithm.
//
// As we true value of the root cannot be located in the presence of noise we instead build a
// piecewise homogeneous probability distribution containing knowledge about it's location.
// Initially this consists of a domain [a, b] which contains the root with a probability
// approaching unity. As more the optimisation proceeds more points are sampled within range [a,b]
// and the probabilities updated.

use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use statrs::{
    distribution::{Continuous, ContinuousCDF},
    statistics::{Max, Min},
};
use std::{fmt, iter, ops::Range};

use super::Sign;
use confidence::ConfidenceLevel;

// How close we allow points in the distribution to be before we consider them to be equal
const EPSILON: f64 = 1e-15;

#[derive(Debug, thiserror::Error)]
pub enum DistributionError {
    #[error("the provided domain must have finite range.")]
    InvalidDomain,
    #[error("trying to insert a sample point which is too close to points already in the domain.")]
    InsertionPointTooClose,
    #[error("the provided point {0} is outside the domain.")]
    PointOutsideDomain(f64),
}

#[derive(Clone, Debug)]
/// The posterior distribution is a piecewise constant probability density function deefined on a
/// fixed domain.
///
/// The probabiliity density function is assumed to be zero outside of the function's domain. This
/// means the cumulative probability density function is zero at domain.start, and one at
/// domain.end.
pub(super) struct Distribution<T> {
    /// The maximum number of sampling points to use in the distribution.
    ///
    /// We pre-allocate elements in the `samples` and `log_probability_density` vectors
    /// so during execution we can avoid reallocating.
    n_max: usize,
    /// The sample nodes of the distribution. This is a vector of length `num_samples`.
    ///
    /// This method retains an ordered vector. The creation of the [`Distribution`] object enters
    /// `domain` endpoints into the vector, which is then of length 2. When the structure is
    /// updated and a new value is inserted the vector is extended by one element, and the new
    /// value is inserted at the relevant point of the vector.
    pub(super) samples: Vec<T>,
    /// The log of the probability density function at each sample point.
    ///
    /// We store the log of the probability density function at each sample point rather than the
    /// PDF itself because in bins where we have very high confidence that the probability is low
    /// can get vecy very small, and in narrow bins can get very large. To avoid overflow of
    /// underflow we store the log.
    pub(super) log_probability_density: Vec<T>,
    /// The range of the domain of the distribution.
    ///
    /// This is the region over which the CDF varies from zero to one. At `domain.start` the CDF is
    /// 0 and at `domain.end` the CDF is 1. This information can also be accessed from the first
    ///   and last elements of the samples vector, but we store here for convenience.
    domain: Range<T>,
}

impl<T> Distribution<T> {
    // Create a new [`Distribution`]
    //
    // This fails if the domain of the distribution is empty.
    //
    // Initially a uniform distribution is assumed on the domain. In future this could be updated
    // to reflect prior knowledge about the location of the root.
    pub(super) fn new(domain: Range<T>, n_max: usize) -> Result<Self, DistributionError>
    where
        T: Copy + PartialOrd + Zero,
    {
        if domain.start >= domain.end {
            return Err(DistributionError::InvalidDomain);
        }
        Ok(Self {
            n_max,
            samples: {
                let mut samples = Vec::with_capacity(n_max + 2);
                samples.push(domain.start);
                samples.push(domain.end);
                samples
            },
            log_probability_density: {
                let mut log_probability_density = Vec::with_capacity(n_max + 1);
                // It has to be in the interval, so the initial probability density is 1: ln(1) = 0
                log_probability_density.push(T::zero());
                log_probability_density
            },
            domain,
        })
    }

    // A distribution is comprised of samples taken at all points in the samples vector.
    //
    // Probabilities are stored in a vector with one fewer elements than the number of points in
    // the samples vector. These repsesent the probability the root lies in the region between the
    // corresponding point in the samples vector.
    //
    // To sum the probabilities we need to weight the probability in each element of the
    // probabilities vector by the fraction of the total domain that that element comprises. This
    // method returns an array containing the fractional widths of each element in the
    // distribution, calculated as the ratio of the element width to the total domain width.
    fn fractional_widths(&self) -> Vec<T>
    where
        T: Float,
    {
        let span = self.domain.end - self.domain.start;
        self.samples
            .windows(2)
            .map(|each| each[1] - each[0])
            .map(|each| each / span)
            .collect()
    }

    pub(super) fn probability(&self) -> Vec<T>
    where
        T: Float,
    {
        self.log_probability_density
            .iter()
            .map(|x| x.exp())
            .zip(self.fractional_widths())
            .map(|(prob, frac)| prob * frac)
            .collect()
    }

    // Find the index of the insertion point for a new sample point
    //
    // As the array is sorted we can just find the index of the first element where the new sample
    // point exceeds a value in the stored samples.
    fn find_insertion_index(&self, x: &T) -> Option<usize>
    where
        T: PartialOrd,
    {
        self.samples.iter().position(|each| x <= each)
    }

    // Insert a new sample point into the distribution
    //
    // After insertion the distribution is sorted, but the probability density function is not
    // consistent. It needs to be updated by the caller
    fn _insert(&mut self, x: T) -> Result<usize, DistributionError>
    where
        T: FromPrimitive + Float + fmt::Debug + ToPrimitive,
    {
        let epsilon = T::from_f64(EPSILON).unwrap();
        if self.samples.iter().any(|each| (*each - x).abs() < epsilon) {
            return Err(DistributionError::InsertionPointTooClose);
        }

        if let Some(index) = self.find_insertion_index(&x) {
            self.samples.insert(index, x);
            // Duplicate the probability density function at the new sample point according to
            let current_probability = self.log_probability_density[index - 1];
            self.log_probability_density
                .insert(index, current_probability);
            return Ok(index);
        }

        // If we didn't find a valid insertion point it means the point is outside the domain of
        // the distribution.
        Err(DistributionError::PointOutsideDomain(x.to_f64().unwrap()))
    }

    // Note here we do not use the implementation from statrs
    //
    // Statrs allows us to find the median using theeir auto-implemenation of the inverse CDF from
    // the [`ContinuousCDF`] trait. This is a numerical solve and is not appropriate for this
    // method for two reasons:
    // 1. We need to find the result to very high accuracy in order to get a sufficiently narrow
    //    confidence interval on the result of the bisection. The numerical method does not give
    //    the result to sufficient accuracy, which means that the bisection method will stagnate.
    // 2. We don't actually need to numerically solve anyway. We have a piecewise constant
    //    function, so an analytical solve is more efficient.
    //
    //  We begin by locating the interval containing the median: we do this by calculating the CDF
    //  at the end of each interval, until we locate the first interval where it exceeds 0.5. At
    //  this point we backtrack and linearly interpolate across the interval to find the median
    //  value.
    pub(super) fn median(&self) -> T
    where
        T: Float + FromPrimitive + fmt::Debug,
    {
        //TODO: Tidy this up

        let span = self.domain.end - self.domain.start;

        let diff = self
            .samples
            .windows(2)
            .map(|each| each[1] - each[0])
            .collect::<Vec<T>>();

        let mut iter = diff
            .into_iter()
            .zip(self.log_probability_density.iter().map(|x| x.exp()));

        let mut last = T::zero();
        let mut this = T::zero();

        let mut last_width = None;
        let mut last_probability = None;
        let half = T::from_f64(0.5).unwrap();

        let mut index = 0;

        for (ii, (width, probability_density)) in iter.by_ref().enumerate() {
            let probability = width * probability_density / span;
            last = this;
            this = this + probability;

            if this > half {
                last_width = Some(width);
                last_probability = Some(probability);
                index = ii;
                break;
            }
        }

        if last_width.is_none() || last_probability.is_none() {
            dbg!(&self.samples);
        }
        let last_width = last_width.unwrap();
        let last_probability = last_probability.unwrap();

        let delta = half - last;

        let x_last = self.samples[index];

        x_last + last_width * delta / last_probability
    }

    pub(super) fn insert(
        &mut self,
        x: T,
        sign_guess: Sign,
        slope: Sign,
        confidence_level: ConfidenceLevel<T>,
    ) -> Result<(), DistributionError>
    where
        T: Float + FromPrimitive + iter::Sum + fmt::Debug,
    {
        if !self.domain.contains(&x) {
            return Err(DistributionError::PointOutsideDomain(x.to_f64().unwrap()));
        }

        // The insertion index is the location of the new sample point in the newly sorted array
        let insertion_index = self._insert(x)?;

        // If this update method is not called after insertion the distribution will be left in an
        // inconsistent state (ie: the probability vector will not sum to unity). The insert method
        // adds a new element to `log_probability_density` but duplicates the value of that element
        // in the prior step
        self.update_probability_density(insertion_index, sign_guess, slope, confidence_level);

        Ok(())
    }

    // The probability density is updated after each insertion. If this method is not called the
    // distribution will be left in an inconsistent state, and the algorithm may not converge to a
    // meaningful value.
    //
    // The density is updated as follows. For a new point `X_m` we calculate the random variable
    // Z_m = sign(Y_m(X_m)). This is the sign of the objective function at the new point, and is
    // either positive or negative one. If the objective is exactly zero we set Z_m = + 1;
    //
    // This evaluation informs us about the root location. If Z_m > 0 then the root is expected to
    // lie to the left of X_m, and if Z_m < 0 it is expected to lie to the right.
    //
    // Querying the region at `X_m` splits the distribution into two regions. The probability mass in
    // the region where the true root x* is expected to be (as indicated by the noisy function
    // evaluation) is increased, while that in regions where it is not expected to be is decreased.
    //
    // Information of the actual observation Y_m(X_m) is discarded: although this could provide
    // more practical information about the location of the root, ignoring it permits a Bayesian
    // motivated update and provides a more robust estimation of the root.
    pub(super) fn update_probability_density(
        &mut self,
        insertion_index: usize,
        sign_guess: Sign,
        slope: Sign,
        confidence_level: ConfidenceLevel<T>,
    ) where
        T: Float + FromPrimitive + iter::Sum + fmt::Debug,
    {
        let half = T::from_f64(0.5).unwrap();
        let confidence_level: T = confidence_level.probability();
        let delta_p =
            -sign_guess.signum::<T>() * slope.signum::<T>() * (confidence_level - half) + half;

        let two = T::one() + T::one();

        // Update following Eq 1.3 / 1.4 in Waeber's thesis
        for ii in 0..insertion_index {
            self.log_probability_density[ii] =
                self.log_probability_density[ii] + (two * (T::one() - delta_p)).ln();
        }
        for ii in insertion_index..self.log_probability_density.len() {
            self.log_probability_density[ii] =
                self.log_probability_density[ii] + (two * delta_p).ln();
        }
    }
}

impl<T: Float> Min<T> for Distribution<T> {
    fn min(&self) -> T {
        self.log_probability_density
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .map(|x| x.exp())
            .unwrap_or(T::zero())
    }
}

impl<T: Float> Max<T> for Distribution<T> {
    fn max(&self) -> T {
        self.log_probability_density
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .map(|x| x.exp())
            .unwrap_or(T::zero())
    }
}

impl<T> Continuous<T, T> for Distribution<T>
where
    T: Float,
{
    fn pdf(&self, x: T) -> T {
        // We need to assume the root can only be inside the domain
        //
        // Faster to check this first before searching for the index in the list
        if !self.domain.contains(&x) {
            return T::zero();
        }

        // We already verified that x is in the domain so we can unwrap
        //
        // The new value would insert ahead of the current value, so we subtract one
        let ii = self.find_insertion_index(&x).unwrap() - 1;

        self.log_probability_density[ii].exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.pdf(x).ln()
    }
}

impl<T> ContinuousCDF<T, T> for Distribution<T>
where
    T: Copy + Float + fmt::Debug,
{
    fn cdf(&self, x: T) -> T {
        if x <= self.domain.start {
            return T::zero();
        }
        if x >= self.domain.end {
            return T::one();
        }

        let ii = self.find_insertion_index(&x).unwrap(); // Unwrap here because we already checked
                                                         // the domain bounds are satisfied

        let mut probability_iter = self.probability().into_iter();
        // Collect all the probability values in intervals before the one containing x
        let cdf_at_interval_start = probability_iter
            .by_ref()
            .take(ii - 1)
            .fold(T::zero(), |acc, each| acc + each);

        // In the region containing x we linearly interpolate the probability density function
        let interval_width = self.samples[ii] - self.samples[ii - 1];
        let cdf_in_interval =
            (x - self.samples[ii - 1]) / interval_width * probability_iter.next().unwrap();

        cdf_at_interval_start + cdf_in_interval
    }
}

#[cfg(test)]
mod tests {
    use super::Distribution;
    use super::Sign;
    use super::{Continuous, ContinuousCDF};
    use confidence::ConfidenceLevel;
    use rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn distribution_has_correct_median_with_no_bisections() {
        let domain = 0.0..10.0;
        let distribution = Distribution::new(domain, 100).unwrap();
        approx::assert_relative_eq!(distribution.median(), 5.0, epsilon = 1e-3);
    }

    #[test]
    fn distribution_rejects_insertion_of_point_outside_domain() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain = 0.0..10.0;
        let mut distribution = Distribution::new(domain.clone(), 100).unwrap();

        let position_of_zero = rng.gen_range(domain.start..domain.end);
        let slope = Sign::Positive;

        let new_point = rng.gen_range(-domain.end..domain.start);

        let sign_guess = if new_point < position_of_zero {
            Sign::Negative
        } else {
            Sign::Positive
        };

        let result = distribution.insert(
            new_point,
            sign_guess,
            slope,
            ConfidenceLevel::ninety_five_percent(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn distribution_rejects_reinsertion_of_point() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain = 0.0..10.0;
        let mut distribution = Distribution::new(domain.clone(), 100).unwrap();

        let position_of_zero = rng.gen_range(domain.start..domain.end);
        let slope = Sign::Positive;

        let num_divisions = 10;

        for _ in 0..num_divisions {
            let new_point = rng.gen_range(domain.start..domain.end);
            let sign_guess = if new_point < position_of_zero {
                Sign::Negative
            } else {
                Sign::Positive
            };

            distribution
                .insert(
                    new_point,
                    sign_guess,
                    slope,
                    ConfidenceLevel::ninety_five_percent(),
                )
                .unwrap();
        }

        let seen_point = distribution.samples[5];
        let sign_guess = if seen_point < position_of_zero {
            Sign::Negative
        } else {
            Sign::Positive
        };

        let result = distribution.insert(
            seen_point,
            sign_guess,
            slope,
            ConfidenceLevel::ninety_five_percent(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn distribution_pdf_is_consistent_with_stored_values() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain = 0.0..10.0;
        let mut distribution = Distribution::new(domain.clone(), 100).unwrap();

        let position_of_zero = rng.gen_range(domain.start..domain.end);
        let slope = Sign::Positive;

        let num_divisions = 10;

        for _ in 0..num_divisions {
            let new_point = rng.gen_range(domain.start..domain.end);
            let sign_guess = if new_point < position_of_zero {
                Sign::Negative
            } else {
                Sign::Positive
            };

            distribution
                .insert(
                    new_point,
                    sign_guess,
                    slope,
                    ConfidenceLevel::ninety_five_percent(),
                )
                .unwrap();
        }

        let num_regions = distribution.log_probability_density.len();

        let num_tests = 20;
        for _ in 0..num_tests {
            let insertion_region = rng.gen_range(0..num_regions);
            let expected = distribution.log_probability_density[insertion_region];

            let point_in_region = rng.gen_range(
                distribution.samples[insertion_region]..distribution.samples[insertion_region + 1],
            );

            let actual = distribution.ln_pdf(point_in_region);

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn distribution_median_is_roughly_consistent_with_inverse_cdf() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain = 0.0..10.0;
        let mut distribution = Distribution::new(domain.clone(), 100).unwrap();

        let position_of_zero = rng.gen_range(domain.start..domain.end);
        let slope = Sign::Positive;

        let num_divisions = 10;

        for _ in 0..num_divisions {
            let new_point = rng.gen_range(domain.start..domain.end);
            let sign_guess = if new_point < position_of_zero {
                Sign::Negative
            } else {
                Sign::Positive
            };

            distribution
                .insert(
                    new_point,
                    sign_guess,
                    slope,
                    ConfidenceLevel::ninety_five_percent(),
                )
                .unwrap();
        }

        let implementation = distribution.median();
        let inverse_cdf = distribution.inverse_cdf(0.5);

        // A large tolerance is used here because the implementation used in statrs is not very
        // accurate
        approx::assert_relative_eq!(implementation, inverse_cdf, epsilon = 1e-3);
    }
}
