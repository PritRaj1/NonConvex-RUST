use nalgebra::DVector;
use crate::limited_memory_bfgs::linesearch::{LineSearch, BacktrackingLineSearch, StrongWolfeLineSearch, HagerZhangLineSearch, MoreThuenteLineSearch, GoldenSectionLineSearch};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::config::{LBFGSConf, LineSearchConf};

pub struct LBFGS<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: LBFGSConf,
    pub opt_prob: OptProb<T, F, G>,
    pub x: DVector<T>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    pub linesearch: Box<dyn LineSearch<T, F, G>>,
    s: Vec<DVector<T>>,
    y: Vec<DVector<T>>,
    has_bounds: bool,
    lower_bounds: Option<DVector<T>>,
    upper_bounds: Option<DVector<T>>,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> LBFGS<T, F, G> {
    pub fn new(conf: LBFGSConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let linesearch: Box<dyn LineSearch<T, F, G>> = match &conf.line_search {
            LineSearchConf::Backtracking(backtracking_conf) => Box::new(BacktrackingLineSearch::new(backtracking_conf)),
            LineSearchConf::StrongWolfe(strong_wolfe_conf) => Box::new(StrongWolfeLineSearch::new(strong_wolfe_conf)),
            LineSearchConf::HagerZhang(hager_zhang_conf) => Box::new(HagerZhangLineSearch::new(hager_zhang_conf)),
            LineSearchConf::MoreThuente(more_thuente_conf) => Box::new(MoreThuenteLineSearch::new(more_thuente_conf)),
            LineSearchConf::GoldenSection(golden_section_conf) => Box::new(GoldenSectionLineSearch::new(golden_section_conf)),
        };

        let f = opt_prob.objective.f(&init_x);
        
        // Check if problem has bounds
        let lower_bounds = opt_prob.objective.x_lower_bound(&init_x);
        let upper_bounds = opt_prob.objective.x_upper_bound(&init_x);
        let has_bounds = lower_bounds.is_some() || upper_bounds.is_some();

        Self { 
            conf, 
            opt_prob, 
            x: init_x.clone(), 
            best_x: init_x.clone(), 
            best_fitness: f.clone(), 
            linesearch,
            s: Vec::new(),
            y: Vec::new(),
            has_bounds,
            lower_bounds,
            upper_bounds,
        }
    }

    fn project_onto_bounds(&self, x: &mut DVector<T>) {
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

    fn compute_cauchy_point(&self, g: &DVector<T>) -> DVector<T> {
        let mut t = T::one();
        let mut x_cp = self.x.clone();
        
        for i in 0..g.len() {
            if g[i] != T::zero() {
                if let (Some(ref lb), Some(ref ub)) = (&self.lower_bounds, &self.upper_bounds) {
                    if g[i] < T::zero() {
                        t = t.min((ub[i] - self.x[i]) / g[i]);
                    } else {
                        t = t.min((lb[i] - self.x[i]) / g[i]);
                    }
                }
            }
        }
        
        x_cp -= g * t;
        self.project_onto_bounds(&mut x_cp);
        x_cp
    }

    pub fn step(&mut self) {
        let g = self.opt_prob.objective.gradient(&self.x).unwrap();
        
        if self.has_bounds {
            self.step_with_bounds(&g); // L-BFGS-B
        } else {
            self.step_without_bounds(&g); // L-BFGS
        }
    }

    fn step_with_bounds(&mut self, g: &DVector<T>) {
        let x_cp = self.compute_cauchy_point(g);
        let mut p = x_cp - &self.x;
        
        let r = self.compute_reduced_gradient(g);
        let z = self.two_loop_recursion(&r);

        self.update_search_direction(&mut p, &z);

        let alpha = self.linesearch.search(&self.x, &p, self.best_fitness, g, &self.opt_prob);
        let mut x_new = &self.x + &p * alpha;
        self.project_onto_bounds(&mut x_new);

        self.update_s_y_vectors(&x_new, g);
        self.update_best_solution(&x_new);
        self.x = x_new;
    }

    fn step_without_bounds(&mut self, g: &DVector<T>) {
        let p = self.compute_search_direction(g);
        
        let alpha = self.linesearch.search(&self.x, &p, self.best_fitness, g, &self.opt_prob);
        
        let x_new = &self.x + &p * alpha;
        
        self.update_s_y_vectors(&x_new, g);
        self.update_best_solution(&x_new);
        self.x = x_new;
    }

    fn compute_reduced_gradient(&self, g: &DVector<T>) -> DVector<T> {
        let mut r = DVector::zeros(self.x.len());
        for i in 0..self.x.len() {
            if !self.is_at_bound(i) {
                r[i] = g[i];
            }
        }
        r
    }

    // Two-loop recursion to approximate the inverse Hessian
    fn two_loop_recursion(&self, r: &DVector<T>) -> DVector<T> {
        let m = self.conf.common.memory_size;
        let mut q = r.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];
        
        for i in (0..self.s.len()).rev() {
            rho[i] = T::one() / self.s[i].dot(&self.y[i]);
            alpha[i] = rho[i] * self.s[i].dot(&q);
            q -= &self.y[i] * alpha[i];
        }
        
        let mut z = q.clone();
        for i in 0..self.s.len() {
            let beta = rho[i] * self.y[i].dot(&z);
            z += &self.s[i] * (alpha[i] - beta);
        }
        z
    }

    fn update_search_direction(&self, p: &mut DVector<T>, z: &DVector<T>) {
        for i in 0..self.x.len() {
            if !self.is_at_bound(i) {
                p[i] = z[i];
            }
        }
    }

    fn compute_search_direction(&self, g: &DVector<T>) -> DVector<T> {
        let m = self.conf.common.memory_size;
        let mut q = g.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];
        
        for i in (0..m).rev() {
            if i < self.s.len() {
                rho[i] = T::one() / self.y[i].dot(&self.s[i]);
                alpha[i] = rho[i] * self.s[i].dot(&q);
                q -= &self.y[i] * alpha[i];
            }
        }
        
        if !self.s.is_empty() {
            let gamma = self.s.last().unwrap().dot(self.y.last().unwrap()) / self.y.last().unwrap().dot(self.y.last().unwrap());
            q *= gamma;
        }
        
        let mut p = q.clone();
        for i in 0..m {
            if i < self.s.len() {
                let beta = rho[i] * self.y[i].dot(&p);
                p += &self.s[i] * (alpha[i] - beta);
            }
        }
        p
    }

    fn update_s_y_vectors(&mut self, x_new: &DVector<T>, g: &DVector<T>) {
        let s_new = x_new - &self.x;
        let y_new = self.opt_prob.objective.gradient(x_new).unwrap() - g;

        if self.s.len() == self.conf.common.memory_size {
            self.s.remove(0);
            self.y.remove(0);
        }
        self.s.push(s_new);
        self.y.push(y_new);
    }

    fn update_best_solution(&mut self, x_new: &DVector<T>) {
        let f_new = self.opt_prob.objective.f(x_new);
        if f_new > self.best_fitness {
            self.best_fitness = f_new;
            self.best_x = x_new.clone();
        }
    }

    fn is_at_bound(&self, i: usize) -> bool {
        let at_lower = self.lower_bounds.as_ref().map_or(false, |lb| self.x[i] == lb[i]);
        let at_upper = self.upper_bounds.as_ref().map_or(false, |ub| self.x[i] == ub[i]);
        at_lower || at_upper
    }
}