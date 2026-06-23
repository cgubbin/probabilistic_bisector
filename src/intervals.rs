use num_traits::{Float, FromPrimitive};

use crate::{
    Sign,
    distribution::{PosteriorDistribution, PosteriorError},
};
use confi::ConfidenceLevel;

pub(crate) trait MeetSemiLattice: Sized {
    type Error;

    fn meet(self, other: Self) -> Result<Self, Self::Error>;
}

#[derive(Debug, Clone, Copy)]
pub struct Interval<T> {
    pub lower: T,
    pub upper: T,
}

impl<T> MeetSemiLattice for Interval<T>
where
    T: Float + Copy + std::fmt::Debug,
{
    type Error = IntervalError<T>;

    fn meet(self, other: Self) -> Result<Self, Self::Error> {
        let l = self.lower.max(other.lower);
        let r = self.upper.min(other.upper);

        if l > r {
            return Err(IntervalError::MeetFailure {
                left: self,
                right: other,
            });
        }

        Ok(Interval { lower: l, upper: r })
    }
}

impl<T> Interval<T> {
    fn new(g: &[usize], knots: &[T]) -> Result<Self, IntervalError<T>>
    where
        T: Copy,
    {
        let start = *g.first().ok_or(IntervalError::EmptyHull)?;
        let end = *g.last().ok_or(IntervalError::EmptyHull)?;

        Ok(Self {
            lower: knots[start],
            upper: knots[end + 1],
        })
    }
}

#[derive(thiserror::Error, Debug)]
enum BisectionError<T> {
    Interval(#[from] IntervalError<T>),
    Posterior(#[from] PosteriorError<T>),
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum IntervalError<T> {
    MeetFailure {
        left: Interval<T>,
        right: Interval<T>,
    },
    EmptyHull,
}

// #[derive(Clone, Debug)]
// pub struct InferenceState<T> {
//     pub n: usize,
//     pub max_n: usize,
//     pub posterior: PosteriorDistribution<T>,
//     pub support: SupportSet<T>,
//     pub confidence: ConfidenceElement<T>,
// }

// impl<T> InferenceState<T> {
//     fn observe(
//         &mut self,
//         x: T,
//         sign: Sign,
//         conf: ConfidenceLevel<T>,
//     ) -> Result<(), BisectionError<T>>
//     where
//         T: Float + FromPrimitive + std::ops::AddAssign + std::iter::Sum<T>,
//     {
//         // 1. posterior (semiring update)
//         self.posterior.observe(x, sign, conf)?;

//         // 2. support (threshold projection)
//         self.support.recompute(&self.posterior);

//         // 3. confidence (lattice meet)
//         let candidate = compute_snapshot(&self.posterior, conf, self.n)?;
//         self.confidence = self.confidence.meet(candidate)?;

//         Ok(())
//     }
// }

// #[derive(Clone, Debug)]
// pub struct SupportSet<T> {
//     pub active_intervals: Vec<(T, T)>,
// }

// impl<T: Float> SupportSet<T> {
//     pub fn recompute(&mut self, posterior: &PosteriorDistribution<T>)
//     where
//         T: Float + FromPrimitive,
//     {
//         // epsilon chosen in probability space
//         let eps = T::from_f64(1e-12).unwrap();
//         let log_eps = eps.ln();

//         self.active_intervals.clear();

//         let mut i = 0;

//         while i < posterior.log_interval_mass.len() {
//             let lp = posterior.log_interval_mass[i];

//             if lp > log_eps {
//                 let start = i;

//                 // extend contiguous support region
//                 while i < posterior.log_interval_mass.len()
//                     && posterior.log_interval_mass[i] > log_eps
//                 {
//                     i += 1;
//                 }

//                 let end = i - 1;

//                 self.active_intervals
//                     .push((posterior.knots[start], posterior.knots[end + 1]));
//             } else {
//                 i += 1;
//             }
//         }
//     }
// }

// #[derive(Debug, Clone, Copy)]
// pub struct ConfidenceElement<T>(Interval<T>);

// impl<T: Float> ConfidenceElement<T> {
//     /// Meet operation: intersection
//     pub fn meet(self, other: Interval<T>) -> Result<Self, IntervalError<T>> {
//         let l = self.0.lower.max(other.lower);
//         let r = self.0.upper.min(other.upper);

//         if l > r {
//             return Err(IntervalError::EmptyMeet {
//                 left: self.0,
//                 right: other,
//             });
//         }

//         Ok(Self(Interval { lower: l, upper: r }))
//     }
// }

// pub fn compute_snapshot<T>(
//     posterior: &PosteriorDistribution<T>,
//     confidence: ConfidenceLevel<T>,
//     n: usize,
// ) -> Result<Interval<T>, IntervalError<T>>
// where
//     T: Float + FromPrimitive,
// {
//     // ----------------------------
//     // Waeber constants
//     // ----------------------------
//     let alpha = confidence.significance().into_inner();
//     let n1 = T::from_usize(n + 1).unwrap();

//     let c = confidence.into_inner();
//     let one_minus_c = T::one() - c;
//     let two = T::one() + T::one();

//     let d = c * (two * c).ln() + one_minus_c * (two * one_minus_c).ln();

//     let beta = (c / one_minus_c).ln();

//     let b = n1 * d - n1.sqrt() * (-(T::one() / two) * (alpha / two).ln()).sqrt() * beta;

//     // ----------------------------
//     // numerical stabilisation
//     // ----------------------------
//     let max_log = posterior
//         .log_interval_mass
//         .iter()
//         .cloned()
//         .fold(T::neg_infinity(), T::max);

//     let b_shifted = b - max_log;

//     // ----------------------------
//     // G_n construction (lattice support set)
//     // ----------------------------
//     let mut g: Vec<usize> = posterior
//         .log_interval_mass
//         .iter()
//         .enumerate()
//         .filter(|(_, lp)| **lp - max_log > b_shifted)
//         .map(|(i, _)| i)
//         .collect();

//     let start = *g.first().unwrap();
//     let end = *g.last().unwrap();

//     let candidate = Interval::new(&g, &posterior.knots)?;

//     Ok(candidate)
// }
// // fn compute_snapshot<T>(
// //     posterior: &PosteriorDistribution<T>,
// //     confidence_level: ConfidenceLevel<T>,
// // ) -> Result<ConfidenceSnapshot<T>, ConfidenceError>
// // where
// //     T: Float + FromPrimitive,
// // {
// //     // ---- Precompute constants used in Waeber-style thresholding ----
// //     //
// //     // These constants encode:
// //     // - confidence level scaling
// //     // - significance correction
// //     // - KL-like adjustment term (d)
// //     // - likelihood ratio shift (beta)
// //     //
// //     // They are iteration-dependent only through `n1` and significance scaling.
// //     let c = confidence_level.into_inner();
// //     let one_minus_c = T::one() - c;

// //     let alpha = confidence_level.significance().into_inner();
// //     let n1 = T::from_usize(self.n + 1).unwrap();

// //     let two = T::one() + T::one();

// //     let d = c * (two * c).ln() + one_minus_c * (two * one_minus_c).ln();

// //     let beta = (c / one_minus_c).ln();

// //     // ---- Threshold construction (Eq 3.7 in Waeber-style derivation) ----
// //     //
// //     // This threshold selects "high posterior mass intervals" in log-space.
// //     // Conceptually: keep indices where posterior mass exceeds a dynamic bound.
// //     let b = n1 * d - n1.sqrt() * (-(T::one() / two) * (alpha / two).ln()).sqrt() * beta;

// //     let b_shifted = b - posterior.max_log_interval_mass();

// //     // ---- High-probability index set ----
// //     //
// //     // G_n = indices where posterior log-density exceeds threshold b
// //     //
// //     // This set defines the support of the confidence interval.
// //     let g: Vec<usize> = posterior
// //         .shifted_log_interval_mass()
// //         .enumerate()
// //         .filter(|(_, lp)| *lp > b_shifted)
// //         .map(|(i, _)| i)
// //         .collect();

// //     let interval = WaeberInterval::new(&g, &posterior.knots)?;

// //     let sequential = SequentialInterval::new(
// //         interval.clone(),
// //         self.values.last().map(|value| &value.sequential),
// //     )?;

// //     Ok(ConfidenceSnapshot {
// //         interval,
// //         sequential,
// //     })
// // }
