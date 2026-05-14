use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};

pub mod algorithms;
pub mod utils;

pub use algorithms::adam::{Adam, AdamConf};
pub use algorithms::cem::{CEMConf, CEM};
pub use algorithms::cma_es::{CMAESConf, CMAES};
pub use algorithms::continuous_genetic::{CGAConf, CGA};
pub use algorithms::differential_evolution::{DEConf, DE};
pub use algorithms::grasp::{GRASPConf, GRASP};
pub use algorithms::limited_memory_bfgs::{LBFGSConf, LBFGS};
pub use algorithms::multi_swarm::{MSPOConf, MSPO};
pub use algorithms::nelder_mead::{NelderMead, NelderMeadConf};
pub use algorithms::parallel_tempering::{PTConf, PT};
pub use algorithms::sg_ascent::{SGAConf, SGAscent};
pub use algorithms::simulated_annealing::{SAConf, SimulatedAnnealing};
pub use algorithms::tabu_search::{TabuConf, TabuSearch};
pub use algorithms::tpe::{TPEConf, TPE};
pub use utils::config::{AlgConf, Config, OptConf};
pub use utils::opt_prob::{
    BooleanConstraintFunction, FloatNumber, ObjectiveFunction, OptProb, OptimizationAlgorithm,
    State,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    AbsoluteTolerance,
    RelativeTolerance,
    Stagnation,
}

pub struct NonConvexOpt<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<bool, N>: Send + Sync,
    OVector<bool, D>: Send + Sync,
    OMatrix<bool, U1, N>: Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, U1, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N>
        + Allocator<N, D>
        + Allocator<D, D>
        + Allocator<U1, D>
        + Allocator<U1, N>,
{
    pub alg: Box<dyn OptimizationAlgorithm<T, N, D>>,
    pub conf: OptConf,
    pub converged: bool,
    pub convergence_reason: Option<ConvergenceReason>,
    best_fitness_history: Vec<T>,
}

impl<T, N, D> NonConvexOpt<T, N, D>
where
    T: FloatNumber + nalgebra::RealField + std::iter::Sum,
    N: Dim,
    D: Dim + nalgebra::DimSub<nalgebra::Const<1>>,
    OVector<bool, N>: Send + Sync,
    OVector<bool, D>: Send + Sync,
    OMatrix<bool, U1, N>: Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, U1, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N>
        + Allocator<N, D>
        + Allocator<D, D>
        + Allocator<U1, D>
        + Allocator<U1, N>
        + Allocator<<D as nalgebra::DimSub<nalgebra::Const<1>>>::Output>,
{
    pub fn new<
        F: ObjectiveFunction<T, D> + 'static,
        G: BooleanConstraintFunction<T, D> + 'static,
    >(
        conf: Config,
        init_pop: OMatrix<T, N, D>,
        obj_f: F,
        constr_f: Option<G>,
        seed: u64,
    ) -> Self {
        let opt_prob = OptProb::new(Box::new(obj_f), constr_f.map(|c| Box::new(c) as _));

        let oc = &conf.opt_conf;
        let alg: Box<dyn OptimizationAlgorithm<T, N, D>> = match conf.alg_conf {
            AlgConf::Adam(c) => Box::new(Adam::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::CEM(c) => Box::new(CEM::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::CGA(c) => Box::new(CGA::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::CMAES(c) => Box::new(CMAES::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::DE(c) => Box::new(DE::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::GRASP(c) => Box::new(GRASP::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::LBFGS(c) => Box::new(LBFGS::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::MSPO(c) => Box::new(MSPO::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::NM(c) => Box::new(NelderMead::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::PT(c) => Box::new(PT::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::SA(c) => Box::new(SimulatedAnnealing::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::SGA(c) => Box::new(SGAscent::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::TPE(c) => Box::new(TPE::new(c, init_pop, opt_prob, oc, seed)),
            AlgConf::TS(c) => Box::new(TabuSearch::new(c, init_pop, opt_prob, oc, seed)),
        };

        Self {
            alg,
            conf: conf.opt_conf,
            converged: false,
            convergence_reason: None,
            best_fitness_history: Vec::new(),
        }
    }

    fn check_convergence(&self, current: T, previous: T) -> Option<ConvergenceReason> {
        let atol: T = T::cast(self.conf.atol);
        let rtol: T = T::cast(self.conf.rtol);
        let min_iter_for_rtol =
            (self.conf.max_iter as f64 * self.conf.rtol_max_iter_fraction).floor() as usize;
        let iter = self.alg.state().iter;
        let eps: T = T::cast(1e-10);

        let abs_imp = num_traits::Float::abs(current - previous);

        if abs_imp < atol && iter > min_iter_for_rtol {
            return Some(ConvergenceReason::AbsoluteTolerance);
        }

        let cur_abs = num_traits::Float::abs(current);
        let rel_converged = if cur_abs > eps {
            abs_imp / cur_abs <= rtol
        } else {
            abs_imp <= atol
        };
        if rel_converged && iter > min_iter_for_rtol {
            return Some(ConvergenceReason::RelativeTolerance);
        }

        if self.best_fitness_history.len() >= self.conf.stagnation_window
            && iter > min_iter_for_rtol
        {
            let oldest = self.best_fitness_history
                [self.best_fitness_history.len() - self.conf.stagnation_window];
            let stag_imp = num_traits::Float::abs(current - oldest);
            let stagnant = stag_imp < atol || (cur_abs > eps && stag_imp / cur_abs <= rtol);
            if stagnant {
                return Some(ConvergenceReason::Stagnation);
            }
        }

        None
    }

    pub fn step(&mut self) {
        if self.converged {
            return;
        }
        let prev = self.alg.state().best_f;
        self.alg.step();
        let cur = self.alg.state().best_f;
        self.best_fitness_history.push(cur);

        let max_history = self.conf.stagnation_window * 2;
        if self.best_fitness_history.len() > max_history {
            let excess = self.best_fitness_history.len() - max_history;
            self.best_fitness_history.drain(0..excess);
        }

        if let Some(reason) = self.check_convergence(cur, prev) {
            self.convergence_reason = Some(reason);
            self.converged = true;
        }
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

    pub fn get_pt_replica_populations(&self) -> Option<Vec<OMatrix<T, N, D>>> {
        self.alg.get_replica_populations()
    }

    pub fn get_pt_replica_temperatures(&self) -> Option<Vec<T>> {
        self.alg.get_replica_temperatures()
    }
}
