use super::{ObservationLocation, PosteriorDistribution};

use num_traits::Float;

impl<T> PosteriorDistribution<T> {
    /// Computes the cumulative distribution function (CDF) at a point `x`.
    ///
    /// Returns:
    ///
    /// ```text
    /// F(x) = P(X ≤ x)
    /// ```
    ///
    /// ## Model assumption
    ///
    /// The density is piecewise constant over each interval `[x_i, x_{i+1})`.
    ///
    /// Therefore the CDF is piecewise linear.
    ///
    /// ## Behavior
    ///
    /// - If `x < knots[0]`: returns 0
    /// - If `x ≥ knots[last]`: returns 1
    /// - Otherwise:
    ///     1. locate interval containing `x`
    ///     2. sum full interval masses below it
    ///     3. linearly interpolate within the interval
    ///
    /// ## Complexity
    ///
    /// O(n) unless interval indexing is optimized
    pub fn cumulative_mass(&self, x: T) -> T
    where
        T: Float,
    {
        let first = self.knots[0];
        let last = self.knots[self.knots.len() - 1];

        match self.locate(x) {
            Ok(ObservationLocation::Boundary) => unreachable!("interior bounds already checked"),

            Ok(ObservationLocation::ExistingKnot(i)) => self
                .log_interval_mass
                .iter()
                .take(i)
                .map(|lp| lp.exp())
                .fold(T::zero(), |acc, p| acc + p),

            Ok(ObservationLocation::Interior(i)) => {
                let left_mass = self
                    .log_interval_mass
                    .iter()
                    .take(i)
                    .map(|lp| lp.exp())
                    .fold(T::zero(), |acc, p| acc + p);

                let interval_mass = self.log_interval_mass[i].exp();
                let left = self.knots[i];
                let right = self.knots[i + 1];

                let frac = (x - left) / (right - left);

                left_mass + frac * interval_mass
            }
            Err(_) => {
                if x < self.knots[0] {
                    T::zero()
                } else {
                    T::one()
                }
            }
        }
    }

    /// Computes the quantile function (inverse CDF).
    ///
    /// Returns `x` such that:
    ///
    /// ```text
    /// P(X ≤ x) = p
    /// ```
    ///
    /// ## Preconditions
    ///
    /// - `p ∈ [0, 1]`
    ///
    /// ## Algorithm
    ///
    /// 1. compute cumulative masses
    /// 2. find interval where CDF crosses `p`
    /// 3. invert linear interpolation inside that interval
    ///
    /// ## Properties
    ///
    /// - Monotone increasing
    /// - Left-continuous inverse of CDF
    ///
    /// ## Complexity
    ///
    /// O(n) (can be improved with binary search or cached prefix sums)
    pub fn quantile(&self, p: T) -> T
    where
        T: Float,
    {
        debug_assert!(p >= T::zero() && p <= T::one());

        if p <= T::zero() {
            return self.knots[0];
        }

        if p >= T::one() {
            return self.knots[self.knots.len() - 1];
        }

        let mut acc = T::zero();

        for i in 0..self.log_interval_mass.len() {
            let mass = self.log_interval_mass[i].exp();
            let next = acc + mass;

            if p <= next {
                let local = (p - acc) / mass;

                let left = self.knots[i];
                let right = self.knots[i + 1];

                return left + local * (right - left);
            }

            acc = next;
        }

        self.knots[self.knots.len() - 1]
    }

    /// Returns the median of the posterior distribution.
    ///
    /// This is the 0.5-quantile:
    ///
    /// ```text
    /// median = Q(0.5)
    /// ```
    ///
    /// ## Properties
    ///
    /// - Robust to skewed distributions
    /// - Defined even when density is discontinuous
    pub fn median(&self) -> T
    where
        T: Float,
    {
        self.quantile(T::from(0.5).unwrap())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn cumulative_mass_is_monotone() {
//         let dist = PosteriorDistribution::new(0.0, 1.0, 20).unwrap();

//         let xs = (0..100).map(|i| i as f64 / 100.0);

//         let mut prev = 0.0;
//         for x in xs {
//             let c = dist.cumulative_mass(x);
//             assert!(c >= prev);
//             prev = c;
//         }
//     }

//     #[test]
//     fn cumulative_mass_boundary_conditions() {
//         let dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

//         assert_eq!(dist.cumulative_mass(0.0), 0.0);
//         assert_eq!(dist.cumulative_mass(1.0), 1.0);
//     }

//     #[test]
//     fn cumulative_mass_out_of_domain() {
//         let dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

//         assert_eq!(dist.cumulative_mass(-1.0), 0.0);
//         assert_eq!(dist.cumulative_mass(2.0), 1.0);
//     }

//     #[test]
//     fn median_lies_in_unit_interval() {
//         let dist = PosteriorDistribution::new(0.0, 1.0, 10).unwrap();

//         let m = dist.median();

//         assert!(m >= 0.0 && m <= 1.0);
//     }
// }
