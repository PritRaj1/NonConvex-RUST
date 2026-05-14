use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use std::collections::VecDeque;

use crate::utils::config::{LBFGSConf, LineSearchConf, OptConf};
use crate::utils::opt_prob::{FloatNumber, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::limited_memory_bfgs::linesearch::{
    BacktrackingLineSearch, GoldenSectionLineSearch, LineSearch, StrongWolfeLineSearch,
};

pub struct LBFGS<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: LBFGSConf,
    pub opt_prob: OptProb<T, D>,
    x: OVector<T, D>,
    pub st: State<T, N, D>,
    linesearch: Box<dyn LineSearch<T, D> + Send + Sync>,

    s: VecDeque<OVector<T, D>>,
    y: VecDeque<OVector<T, D>>,

    has_bounds: bool,
    lower_bounds: Option<OVector<T, D>>,
    upper_bounds: Option<OVector<T, D>>,

    current_memory_size: usize,

    stagnation_counter: usize,
    last_restart_iter: usize,
    last_improvement: T,
    success_history: VecDeque<bool>,
    improvement_history: VecDeque<f64>,
}

impl<T, N, D> LBFGS<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(
        conf: LBFGSConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        _opt_conf: &OptConf,
        _seed: u64,
    ) -> Self {
        let st = State::from_seed(init_pop, &opt_prob);
        let best_f = st.best_f;

        let linesearch: Box<dyn LineSearch<T, D> + Send + Sync> = match &conf.line_search {
            LineSearchConf::Backtracking(c) => Box::new(BacktrackingLineSearch::new(c)),
            LineSearchConf::StrongWolfe(c) => Box::new(StrongWolfeLineSearch::new(c)),
            LineSearchConf::GoldenSection(c) => Box::new(GoldenSectionLineSearch::new(c)),
        };

        let lower_bounds = opt_prob.objective.x_lower_bound(&st.best_x);
        let upper_bounds = opt_prob.objective.x_upper_bound(&st.best_x);
        let has_bounds = lower_bounds.is_some() || upper_bounds.is_some();
        let current_memory_size = conf.common.memory_size;

        Self {
            x: st.best_x.clone(),
            st,
            opt_prob,
            conf: conf.clone(),
            linesearch,
            s: VecDeque::with_capacity(current_memory_size),
            y: VecDeque::with_capacity(current_memory_size),
            has_bounds,
            lower_bounds,
            upper_bounds,
            current_memory_size,
            stagnation_counter: 0,
            last_restart_iter: 0,
            last_improvement: best_f,
            success_history: VecDeque::with_capacity(conf.advanced.success_history_size),
            improvement_history: VecDeque::with_capacity(conf.advanced.improvement_history_size),
        }
    }

    fn project_onto_bounds(&self, x: &mut OVector<T, D>) {
        if let Some(ref lb) = self.lower_bounds {
            for i in 0..x.len() {
                x[i] = x[i].max(lb[i]);
            }
        }
        if let Some(ref ub) = self.upper_bounds {
            for i in 0..x.len() {
                x[i] = x[i].min(ub[i]);
            }
        }
    }

    // |x - bound| < eps; exact float equality misses rounded points
    fn is_at_bound(&self, i: usize) -> bool {
        let eps = T::cast(1e-12);
        let xi = self.x[i];
        let lo = self
            .lower_bounds
            .as_ref()
            .is_some_and(|lb| num_traits::Float::abs(xi - lb[i]) <= eps);
        let hi = self
            .upper_bounds
            .as_ref()
            .is_some_and(|ub| num_traits::Float::abs(xi - ub[i]) <= eps);
        lo || hi
    }

    // L-BFGS two-loop recursion (Nocedal & Wright Alg 7.4); returns H_k · q
    // for ascent on max problems we use +z (no negation at the end)
    fn two_loop_recursion(&self, q_in: &OVector<T, D>) -> OVector<T, D> {
        let m = self.s.len();
        let mut q = q_in.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];
        let cond_thresh = T::cast(self.conf.advanced.numerical_safeguards.conditioning_threshold);

        for i in (0..m).rev() {
            let s_dot_y = self.s[i].dot(&self.y[i]);
            if num_traits::Float::abs(s_dot_y) < cond_thresh {
                continue;
            }
            rho[i] = T::one() / s_dot_y;
            alpha[i] = rho[i] * self.s[i].dot(&q);
            q -= &self.y[i] * alpha[i];
        }

        // initial Hessian scaling γ = sᵀy / yᵀy (latest pair)
        let mut z = if let (Some(s_last), Some(y_last)) = (self.s.back(), self.y.back()) {
            let denom = y_last.dot(y_last);
            if denom > T::zero() {
                q * (s_last.dot(y_last) / denom)
            } else {
                q
            }
        } else {
            q
        };

        for i in 0..m {
            if rho[i] == T::zero() {
                continue;
            }
            let beta = rho[i] * self.y[i].dot(&z);
            z += &self.s[i] * (alpha[i] - beta);
        }
        z
    }

    // bounded path uses gradient-projection step + two-loop for the free coords
    fn step_with_bounds(&mut self, g: &OVector<T, D>) {
        // freeze coords already on a bound that gradient would push past
        let mut r = OVector::<T, D>::zeros_generic(D::from_usize(self.x.len()), U1);
        for i in 0..self.x.len() {
            if !self.is_at_bound(i) {
                r[i] = g[i];
            }
        }
        let z = self.two_loop_recursion(&r);

        let mut p = z.clone();
        for i in 0..p.len() {
            if self.is_at_bound(i) {
                p[i] = T::zero();
            }
        }

        let alpha =
            self.linesearch
                .search(&self.st.best_x, &p, self.st.best_f, g, &self.opt_prob);
        let mut x_new = &self.x + &p * alpha;
        self.project_onto_bounds(&mut x_new);

        self.update_s_y(&x_new, g);
        self.update_best(&x_new);
        self.x = x_new;
    }

    fn step_without_bounds(&mut self, g: &OVector<T, D>) {
        let p = self.two_loop_recursion(g);
        let alpha =
            self.linesearch
                .search(&self.st.best_x, &p, self.st.best_f, g, &self.opt_prob);
        let x_new = &self.x + &p * alpha;
        self.update_s_y(&x_new, g);
        self.update_best(&x_new);
        self.x = x_new;
    }

    fn update_s_y(&mut self, x_new: &OVector<T, D>, g: &OVector<T, D>) {
        let s_new = x_new - &self.st.best_x;
        let y_new = self.opt_prob.objective.gradient(x_new).unwrap() - g;
        let curvature = s_new.dot(&y_new);
        if curvature > T::cast(self.conf.advanced.numerical_safeguards.curvature_threshold) {
            if self.s.len() == self.current_memory_size {
                self.s.pop_front();
                self.y.pop_front();
            }
            self.s.push_back(s_new);
            self.y.push_back(y_new);
        }
    }

    fn update_best(&mut self, x_new: &OVector<T, D>) {
        let f_new = self.opt_prob.evaluate(x_new);
        if f_new > self.st.best_f {
            let improvement = f_new - self.st.best_f;
            self.last_improvement = f_new;
            self.st.best_f = f_new;
            self.st.best_x = x_new.clone();
            self.success_history.push_back(true);
            self.improvement_history
                .push_back(improvement.to_f64().unwrap_or(0.0));
            self.stagnation_counter = 0;
        } else {
            self.success_history.push_back(false);
            self.improvement_history.push_back(0.0);
            self.stagnation_counter += 1;
        }
        if self.success_history.len() > self.conf.advanced.success_history_size {
            self.success_history.pop_front();
        }
        if self.improvement_history.len() > self.conf.advanced.improvement_history_size {
            self.improvement_history.pop_front();
        }
    }

    fn adapt_parameters(&mut self) {
        if !self.conf.advanced.adaptive_parameters {
            return;
        }
        if self.success_history.len() < 5 {
            return;
        }
        let success_rate = self.success_history.iter().filter(|&&x| x).count() as f64
            / self.success_history.len() as f64;

        if self.conf.advanced.memory_adaptation.adaptive_memory {
            if success_rate < 0.2 {
                self.current_memory_size = (self.current_memory_size + 1)
                    .min(self.conf.advanced.memory_adaptation.max_memory_size);
            } else if success_rate > 0.6 {
                self.current_memory_size = (self.current_memory_size.saturating_sub(1))
                    .max(self.conf.advanced.memory_adaptation.min_memory_size);
            }
        }
    }

    fn check_restart(&mut self) -> bool {
        match &self.conf.advanced.restart_strategy {
            super::config::RestartStrategy::None => false,
            super::config::RestartStrategy::Periodic { frequency } => {
                self.st.iter - self.last_restart_iter >= *frequency
            }
            super::config::RestartStrategy::Stagnation {
                max_iterations,
                threshold,
            } => {
                self.stagnation_counter >= *max_iterations
                    || self.last_improvement.to_f64().unwrap_or(0.0) < *threshold
            }
            super::config::RestartStrategy::Adaptive {
                base_frequency,
                adaptation_rate,
            } => {
                let adaptive_frequency = (*base_frequency as f64
                    * (1.0 + adaptation_rate * self.stagnation_counter as f64))
                    as usize;
                self.st.iter - self.last_restart_iter >= adaptive_frequency
            }
        }
    }

    fn perform_restart(&mut self) {
        self.s.clear();
        self.y.clear();
        self.current_memory_size = self.conf.common.memory_size;
        self.stagnation_counter = 0;
        self.last_improvement = self.st.best_f;
        self.last_restart_iter = self.st.iter;
        self.success_history.clear();
        self.improvement_history.clear();
    }

    fn check_stagnation(&self) -> bool {
        let win = self.conf.advanced.stagnation_detection.stagnation_window;
        if self.improvement_history.len() < win {
            return false;
        }
        let avg: f64 = self
            .improvement_history
            .iter()
            .rev()
            .take(win)
            .copied()
            .sum::<f64>()
            / win as f64;
        avg < self.conf.advanced.stagnation_detection.improvement_threshold
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for LBFGS<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        if self.check_restart() {
            self.perform_restart();
        }
        if self.check_stagnation() {
            self.stagnation_counter += 1;
        }

        let g = self.opt_prob.objective.gradient(&self.x).unwrap();

        if self.has_bounds {
            self.step_with_bounds(&g);
        } else {
            self.step_without_bounds(&g);
        }

        let fitness = self.opt_prob.evaluate(&self.x);
        if fitness > self.st.best_f {
            self.st.best_f = fitness;
            self.st.best_x = self.x.clone();
        }
        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = fitness;
        self.st.constraints[0] = self.opt_prob.is_feasible(&self.x);

        self.adapt_parameters();
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}
