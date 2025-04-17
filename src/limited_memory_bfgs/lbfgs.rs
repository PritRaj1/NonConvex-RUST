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

        Self { 
            conf, 
            opt_prob, 
            x: init_x.clone(), 
            best_x: init_x.clone(), 
            best_fitness: f.clone(), 
            linesearch,
            s: Vec::new(),
            y: Vec::new(),
        }
    }

    pub fn step(&mut self) {
        let g = self.opt_prob.objective.gradient(&self.x).unwrap();
        
        let m = self.conf.common.memory_size;
        
        // Search direction using L-BFGS two-loop recursion
        let mut q = g.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];
        
        // First loop - compute alpha and update q
        for i in (0..m).rev() {
            if i < self.s.len() {
                rho[i] = T::one() / self.y[i].dot(&self.s[i]);
                alpha[i] = rho[i] * self.s[i].dot(&q);
                q -= &self.y[i] * alpha[i];
            }
        }
        
        // Scale q
        if !self.s.is_empty() {
            let gamma = self.s.last().unwrap().dot(self.y.last().unwrap()) / self.y.last().unwrap().dot(self.y.last().unwrap());
            q *= gamma;
        }
        
        // Second loop - compute search direction p
        let mut p = q.clone();
        for i in 0..m {
            if i < self.s.len() {
                let beta = rho[i] * self.y[i].dot(&p);
                p += &self.s[i] * (alpha[i] - beta);
            }
        }
        
        let alpha = self.linesearch.search(&self.x, &p, self.best_fitness, &g, &self.opt_prob);
        
        let mut x_new = &self.x + &p * alpha;
        let f_new = self.opt_prob.objective.f(&x_new);
        
        // Project onto feasible set if needed
        if let Some(ref constraints) = self.opt_prob.constraints {
            if !constraints.g(&x_new) {
                if let (Some(lb), Some(ub)) = (self.opt_prob.objective.x_lower_bound(&x_new), 
                                             self.opt_prob.objective.x_upper_bound(&x_new)) {
                    for i in 0..x_new.len() {
                        x_new[i] = x_new[i].max(lb[i]).min(ub[i]);
                    }
                }
            }
        }
        
        let s_new = &x_new - &self.x;
        let y_new = self.opt_prob.objective.gradient(&x_new).unwrap() - g;
        
        if self.s.len() == m {
            self.s.remove(0);
            self.y.remove(0);
        }
        self.s.push(s_new);
        self.y.push(y_new);
        
        let x_new_clone = x_new.clone();
        self.x = x_new;
        
        if f_new > self.best_fitness {
            self.best_fitness = f_new;
            self.best_x = x_new_clone;
        }
    }
}