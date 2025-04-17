use nalgebra::DVector;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::config::{
    BacktrackingConf,
    StrongWolfeConf,
    HagerZhangConf,
    MoreThuenteConf,
    GoldenSectionConf,
};

pub trait LineSearch<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    fn search(
        &self,
        x: &DVector<T>,
        p: &DVector<T>,
        f: T,
        g: &DVector<T>,
        opt_prob: &OptProb<T, F, G>,
    ) -> T;
}

pub struct BacktrackingLineSearch {
    conf: BacktrackingConf,
}

impl BacktrackingLineSearch {
    pub fn new(conf: &BacktrackingConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> LineSearch<T, F, G> for BacktrackingLineSearch {
    fn search(
        &self,
        x: &DVector<T>,
        p: &DVector<T>,
        f: T,
        g: &DVector<T>,
        opt_prob: &OptProb<T, F, G>,
    ) -> T {
        let mut alpha = T::one();
        let mut x_new = x + p * alpha;
        
        // Repeat until the Armijo condition is satisfied
        while opt_prob.objective.f(&x_new) > f + T::from_f64(self.conf.c1).unwrap() * alpha * g.dot(p) {
            alpha = alpha * T::from_f64(self.conf.rho).unwrap();
            x_new = x + p * alpha;
        }
        
        alpha
    }
}

pub struct StrongWolfeLineSearch {
    conf: StrongWolfeConf,
}

impl StrongWolfeLineSearch {
    pub fn new(conf: &StrongWolfeConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> LineSearch<T, F, G> for StrongWolfeLineSearch {
    fn search(
        &self,
        x: &DVector<T>,
        p: &DVector<T>,
        f: T,
        g: &DVector<T>,
        opt_prob: &OptProb<T, F, G>,
    ) -> T {
        let c1 = T::from_f64(self.conf.c1).unwrap();
        let c2 = T::from_f64(self.conf.c2).unwrap();
        let mut alpha = T::one();
        let mut alpha_low = T::zero();
        let mut alpha_high = T::from_f64(10.0).unwrap();
        let initial_gp = g.dot(p);

        for _ in 0..self.conf.max_iters {
            let x_new = x + p * alpha;
            let f_new = opt_prob.objective.f(&x_new);
            let g_new = opt_prob.objective.gradient(&x_new).unwrap();
            
            // Check Armijo condition
            if f_new > f + c1 * alpha * g.dot(p) {
                alpha_high = alpha;
                alpha = (alpha_low + alpha_high) / T::from_f64(2.0).unwrap();
                continue;
            }
            
            // Check strong Wolfe condition
            if g_new.dot(p).abs() > c2 * initial_gp.abs() {
                alpha_low = alpha;
                alpha = if alpha_high == T::from_f64(10.0).unwrap() {
                    alpha * T::from_f64(2.0).unwrap()
                } else {
                    (alpha_low + alpha_high) / T::from_f64(2.0).unwrap()
                };
                continue;
            }
            
            return alpha;
        }
        
        T::one()
    }
}

pub struct HagerZhangLineSearch {
    conf: HagerZhangConf,
}

impl HagerZhangLineSearch {
    pub fn new(conf: &HagerZhangConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> LineSearch<T, F, G> for HagerZhangLineSearch {
    fn search(
        &self,
        x: &DVector<T>,
        p: &DVector<T>,
        f: T,
        g: &DVector<T>,
        opt_prob: &OptProb<T, F, G>,
    ) -> T {
        let c1 = T::from_f64(self.conf.c1).unwrap();
        let c2 = T::from_f64(self.conf.c2).unwrap();
        let theta = T::from_f64(self.conf.theta).unwrap();
        let gamma = T::from_f64(self.conf.gamma).unwrap();
        let mut alpha = T::one();
        let initial_gp = g.dot(p);
        for _ in 0..self.conf.max_iters {
            let x_new = x + p * alpha;
            let f_new = opt_prob.objective.f(&x_new);
            let g_new = opt_prob.objective.gradient(&x_new).unwrap();
            
            // Check Armijo condition
            if f_new > f + c1 * alpha * g.dot(p) {
                alpha = alpha * gamma;
                continue;
            }
            
            // Check approximate Wolfe condition
            let g_new_p = g_new.dot(p);
            if g_new_p.abs() > c2 * initial_gp.abs() {
                let delta = theta * (alpha * (initial_gp - g_new_p)) / (f - f_new + alpha * initial_gp);
                alpha = alpha + delta;
                continue;
            }
            
            return alpha;
        }
        
        T::one()
    }
}

pub struct MoreThuenteLineSearch {
    conf: MoreThuenteConf,
}

impl MoreThuenteLineSearch {
    pub fn new(conf: &MoreThuenteConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> LineSearch<T, F, G> for MoreThuenteLineSearch {
    fn search(
        &self,
        x: &DVector<T>,
        p: &DVector<T>,
        f: T,
        g: &DVector<T>,
        opt_prob: &OptProb<T, F, G>,
    ) -> T {
        let ftol = T::from_f64(self.conf.ftol).unwrap();
        let gtol = T::from_f64(self.conf.gtol).unwrap();
        let mut alpha = T::one();
        let mut alpha_prev = T::zero();
        let initial_gp = g.dot(p);

        let mut f_prev = f;
        let mut g_prev = initial_gp;

        for _ in 0..self.conf.max_iters {
            let x_new = x + p * alpha;
            let f_new = opt_prob.objective.f(&x_new);
            let g_new = opt_prob.objective.gradient(&x_new).unwrap();
            let g_new_p = g_new.dot(p);
            
            // Check both Armijo and strong Wolfe conditions
            if f_new > f + ftol * alpha * initial_gp || g_new_p.abs() > gtol * initial_gp.abs() {
                alpha = self.update_alpha(alpha, alpha_prev, f_new, f_prev, g_new_p, g_prev);
                f_prev = f_new;
                g_prev = g_new_p;
                alpha_prev = alpha;
                continue;
            }

            return alpha;
        }
        
        T::one()
    }
}

impl MoreThuenteLineSearch {
    fn update_alpha<T: FloatNum>(
        &self,
        alpha: T,
        alpha_prev: T,
        f: T,
        f_prev: T,
        g: T,
        g_prev: T,
    ) -> T {
        // Cubic interpolation between the two most recent points
        let d1 = g_prev + g - T::from_f64(3.0).unwrap() * (f_prev - f) / (alpha_prev - alpha);
        let d2_squared = d1 * d1 - g_prev * g;
        
        // Handle case where d2_squared is negative due to numerical issues
        if d2_squared < T::zero() {
            return (alpha + alpha_prev) * T::from_f64(0.5).unwrap(); // Fallback to bisection
        }
        
        let d2 = d2_squared.sqrt();
        let theta = T::from_f64(0.5).unwrap(); // Reduced from 3.0 for more conservative steps
        let delta_alpha = alpha - alpha_prev;
        let delta_g = g - g_prev;
        let two_d2 = T::from_f64(2.0).unwrap() * d2;
        
        let d2_term = if alpha > alpha_prev { g + d2 - d1 } else { g - d2 - d1 };
        alpha - delta_alpha * d2_term / (delta_g + two_d2) * theta
    }
}

pub struct GoldenSectionLineSearch {
    conf: GoldenSectionConf,
}

impl GoldenSectionLineSearch {
    pub fn new(conf: &GoldenSectionConf) -> Self {
        Self { conf: conf.clone() }
    }

    // Helper function to bracket the minimum
    fn bracket_minimum<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>>(
        &self,
        x: &DVector<T>,
        p: &DVector<T>,
        opt_prob: &OptProb<T, F, G>,
    ) -> (T, T, T) {
        let golden_ratio: T = T::from_f64((5.0_f64).sqrt() * 0.5 + 0.5).unwrap();
        let bracket_factor = T::from_f64(self.conf.bracket_factor).unwrap();
        
        let mut a = T::zero();
        let mut b = T::one();
        let mut c = b * golden_ratio;
        
        let mut fa = opt_prob.objective.f(&(x + p * a));
        let mut fb = opt_prob.objective.f(&(x + p * b));
        let mut fc = opt_prob.objective.f(&(x + p * c));
        
        // Expand the bracket until we find a triplet where the middle point is lower
        while fb > fa || fb > fc {
            if fb > fa {
                c = b;
                b = a;
                a = b / bracket_factor;
                fc = fb;
                fb = fa;
                fa = opt_prob.objective.f(&(x + p * a));
            } else {
                a = b;
                b = c;
                c = b * bracket_factor;
                fa = fb;
                fb = fc;
                fc = opt_prob.objective.f(&(x + p * c));
            }
        }
        
        (a, b, c)
    }
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> LineSearch<T, F, G> for GoldenSectionLineSearch {
    fn search(
        &self,
        x: &DVector<T>,
        p: &DVector<T>,
        _f: T,
        _g: &DVector<T>,
        opt_prob: &OptProb<T, F, G>,
    ) -> T {
        let resphi = T::from_f64((3.0_f64 - (5.0_f64).sqrt()) / 2.0).unwrap();
        let tol = T::from_f64(self.conf.tol).unwrap();
        
        let (mut a, b, mut c) = self.bracket_minimum(x, p, opt_prob);
        let mut x0 = b - resphi * (c - a);
        let mut x1 = a + resphi * (c - a);
        let mut f0 = opt_prob.objective.f(&(x + p * x0));
        let mut f1 = opt_prob.objective.f(&(x + p * x1));

        loop {
            if (c - a).abs() < tol {
                break (a + c) / T::from_f64(2.0).unwrap();
            }

            if f0 < f1 {
                c = x1;
                x1 = x0;
                f1 = f0;
                x0 = b - resphi * (c - a);
                f0 = opt_prob.objective.f(&(x + p * x0));
            } else {
                a = x0;
                x0 = x1;
                f0 = f1;
                x1 = a + resphi * (c - a);
                f1 = opt_prob.objective.f(&(x + p * x1));
            }
        }
    }
}