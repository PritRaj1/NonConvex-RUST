use nalgebra::{
    allocator::Allocator, 
    DefaultAllocator, 
    Dim, 
    OVector, 
    OMatrix
};

pub mod algorithms;
pub mod utils;
use crate::utils::config::{Config, AlgConf, OptConf};

use crate::algorithms::{
    continous_ga::cga::CGA,
    parallel_tempering::pt::PT,
    adam::adam::Adam,
    sg_ascent::sga::SGAscent,
    tabu_search::tabu::TabuSearch,
    grasp::grasp::GRASP,
    nelder_mead::nm::NelderMead,
    limited_memory_bfgs::lbfgs::LBFGS,
    multi_swarm::mspo::MSPO,
    simulated_annealing::sa::SimulatedAnnealing,
    differential_evolution::de::DE,
    cma_es::cma_es::CMAES,
};

use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    ObjectiveFunction, 
    BooleanConstraintFunction,
    OptimizationAlgorithm,
    State
};

pub struct Result<T: FloatNum, N: Dim, D: Dim> 
where
    DefaultAllocator: Allocator<D>
                    + Allocator<N, D>
                    + Allocator<N>
{
    pub best_x: OVector<T, D>,
    pub best_f: T,
    pub final_pop: OMatrix<T, N, D>,
    pub final_fitness: OVector<T, N>,
    pub final_constraints: OVector<bool, N>,
    pub convergence_iter: usize,
}

pub struct NonConvexOpt<T: FloatNum, N: Dim, D: Dim> {
    pub alg: Box<dyn OptimizationAlgorithm<T, N, D>>,
    pub conf: OptConf,
    pub converged: bool,
}

impl<T: FloatNum, N: Dim, D: Dim> NonConvexOpt<T, N, D> 
where 
    DefaultAllocator: Allocator<D> 
                    + Allocator<N, D> 
                    + Allocator<N>
{
    pub fn new<F: ObjectiveFunction<T, D> + 'static, G: BooleanConstraintFunction<T, D> + 'static>(
        conf: Config, 
        init_pop: OMatrix<T, N, D>, 
        obj_f: F, 
        constr_f: Option<G>,
    ) -> Self {
        let opt_prob = OptProb::new(
            Box::new(obj_f), 
            match constr_f {
                Some(constr_f) => Some(Box::new(constr_f)),
                None => None,
            }
        );

        let alg: Box<dyn OptimizationAlgorithm<T, N, D>> = match conf.alg_conf {
            AlgConf::CGA(cga_conf) => Box::new(CGA::new(cga_conf, init_pop, opt_prob, conf.opt_conf.max_iter)),
            AlgConf::PT(pt_conf) => Box::new(PT::new(pt_conf, init_pop, opt_prob, conf.opt_conf.max_iter)),
            AlgConf::TS(ts_conf) => Box::new(TabuSearch::new(ts_conf, init_pop, opt_prob)),
            AlgConf::Adam(adam_conf) => Box::new(Adam::new(adam_conf, init_pop.row(0), opt_prob)),
            AlgConf::GRASP(grasp_conf) => Box::new(GRASP::new(grasp_conf, init_pop, opt_prob)),
            AlgConf::SGA(sga_conf) => Box::new(SGAscent::new(sga_conf, init_pop, opt_prob)),
            AlgConf::NM(nm_conf) => Box::new(NelderMead::new(nm_conf, init_pop, opt_prob)),
            AlgConf::LBFGS(lbfgs_conf) => Box::new(LBFGS::new(lbfgs_conf, init_pop, opt_prob)),
            AlgConf::MSPO(mspo_conf) => Box::new(MSPO::new(mspo_conf, init_pop, opt_prob)),
            AlgConf::SA(sa_conf) => Box::new(SimulatedAnnealing::new(sa_conf, init_pop, opt_prob)),
            AlgConf::DE(de_conf) => Box::new(DE::new(de_conf, init_pop, opt_prob)),
            AlgConf::CMAES(cma_es_conf) => Box::new(CMAES::new(cma_es_conf, init_pop, opt_prob)),
        };

        Self { alg, conf: conf.opt_conf, converged: false }
    }

    fn check_convergence(&self, current_best: T, previous_best: T) -> bool {
        let converged = (-current_best).exp() <= T::from_f64(self.conf.atol).unwrap()
            || ((current_best - previous_best).abs() <= T::from_f64(self.conf.rtol).unwrap() && self.alg.state().iter > (self.conf.max_iter as f64 * self.conf.rtol_max_iter_fraction).floor() as usize);
        if converged {
            println!("Converged in {} iterations", self.alg.state().iter);
        }
        
        converged
    }

    pub fn step(&mut self) {
        if self.converged {
            return;
        }

        let previous_best_fitness = self.alg.state().best_f;
        self.alg.step();
        let current_best_fitness = self.alg.state().best_f;

        self.converged = self.check_convergence(
            current_best_fitness, 
            previous_best_fitness
        );
    }

    pub fn run(&mut self) -> &State<T, N, D> {
        while !self.converged && self.alg.state().iter < self.conf.max_iter {
            self.step();
        }
        self.alg.state()
    }

    pub fn get_best_individual(&self) -> OVector<T, D> {
        self.alg.state().best_x.clone()
    }

    pub fn get_population(&self) -> OMatrix<T, N, D> {
        self.alg.state().pop.clone()
    }
}