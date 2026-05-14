use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rayon::prelude::*;

use super::config::{DEConf, DEStrategy, MutationType};
use crate::utils::config::OptConf;
use crate::utils::opt_prob::{FloatNumber, OptProb, OptimizationAlgorithm, State};
use crate::utils::rng;

use crate::algorithms::differential_evolution::{
    bounds::BoundsCache,
    mutation::{Best1Bin, Best2Bin, MutationStrategy, Rand1Bin, Rand2Bin, RandToBest1Bin},
    parameter_adaptation::{
        JADEParameterAdaptation, ParameterAdaptationType, StandardParameterAdaptation,
    },
};

pub struct DE<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    pub conf: DEConf,
    pub st: State<T, N, D>,
    pub opt_prob: OptProb<T, D>,
    parameter_adaptation: ParameterAdaptationType,
    bounds_cache: BoundsCache<T, D>,
    seed: u64,
}

impl<T, N, D> DE<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    pub fn new(
        conf: DEConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        _opt_conf: &OptConf,
        seed: u64,
    ) -> Self {
        let dim = init_pop.ncols();
        let st = State::from_population(init_pop, &opt_prob);

        let parameter_adaptation = match &conf.mutation_type {
            MutationType::Standard(s) => {
                ParameterAdaptationType::Standard(StandardParameterAdaptation::new(s.f, s.cr))
            }
            MutationType::Adaptive(a) if a.use_jade => ParameterAdaptationType::JADE(Box::new(
                JADEParameterAdaptation::new(a.memory_size, seed),
            )),
            MutationType::Adaptive(a) => {
                ParameterAdaptationType::Standard(StandardParameterAdaptation::new(a.f, a.cr))
            }
        };

        Self {
            conf,
            st,
            opt_prob,
            parameter_adaptation,
            bounds_cache: BoundsCache::new(dim),
            seed,
        }
    }

    fn select_trial(
        &self,
        trial_fitness: T,
        trial_constraint: bool,
        current_fitness: T,
        current_constraint: bool,
    ) -> bool {
        match (trial_constraint, current_constraint) {
            (true, true) => {
                // Both feasible - compare fitness with tolerance
                let eps = T::cast(1e-10);
                trial_fitness > current_fitness + eps
            }
            (true, false) => true,  // Prefer feasible
            (false, true) => false, // Keep feasible
            (false, false) => {
                // Both infeasible - compare fitness
                trial_fitness > current_fitness
            }
        }
    }
}

impl<T: FloatNumber, N: Dim, D: Dim> OptimizationAlgorithm<T, N, D> for DE<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let pop_size = self.st.pop.nrows();

        let sample_individual = self.st.pop.row(0).transpose();
        let _bounds = self
            .bounds_cache
            .get_bounds(&self.opt_prob, &sample_individual);

        let seed = self.seed;
        let iter = self.st.iter as u64;
        let trials: Vec<_> = (0..pop_size)
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    rng::split(seed, [iter, thread_id as u64])
                },
                |rng, i| {
                    let strategy = match &self.conf.mutation_type {
                        MutationType::Standard(standard) => &standard.strategy,
                        MutationType::Adaptive(adaptive) => &adaptive.strategy,
                    };

                    let strategy: &dyn MutationStrategy<T, N, D> = match strategy {
                        DEStrategy::Rand1Bin => &Rand1Bin,
                        DEStrategy::Best1Bin => &Best1Bin,
                        DEStrategy::RandToBest1Bin => &RandToBest1Bin,
                        DEStrategy::Best2Bin => &Best2Bin,
                        DEStrategy::Rand2Bin => &Rand2Bin,
                    };

                    // split adaptation RNG per (iter, i) — clone alone shares state
                    let mut local_pa = self.parameter_adaptation.clone();
                    if let ParameterAdaptationType::JADE(jade) = &mut local_pa {
                        jade.rng = rng::split(seed, [iter, i as u64, 0xDEu64]);
                    }
                    let (f, cr) = local_pa.generate_parameters();

                    let trial = strategy.generate_trial(
                        &self.st.pop,
                        Some(&self.st.best_x),
                        i,
                        T::cast(f),
                        T::cast(cr),
                        rng,
                    );

                    let fitness = self.opt_prob.evaluate(&trial);
                    let constraint = self.opt_prob.is_feasible(&trial);

                    let success = self.select_trial(
                        fitness,
                        constraint,
                        self.st.fitness[i],
                        self.st.constraints[i],
                    );

                    (i, trial, fitness, constraint, success, f, cr)
                },
            )
            .collect();

        let updates: Vec<_> = trials
            .into_iter()
            .filter_map(
                |(i, trial, trial_fitness, trial_constraint, success, f, cr)| {
                    if success {
                        self.parameter_adaptation.record_success(f, cr);
                        Some((i, trial, trial_fitness, trial_constraint))
                    } else {
                        None
                    }
                },
            )
            .collect();

        self.parameter_adaptation.update_parameters();

        let mut new_population = self.st.pop.clone();
        let mut new_fitness = self.st.fitness.clone();
        let mut new_constraints = self.st.constraints.clone();

        for (i, trial, trial_fitness, trial_constraint) in updates {
            new_population.set_row(i, &trial.transpose());
            new_fitness[i] = trial_fitness;
            new_constraints[i] = trial_constraint;
        }

        self.st.pop = new_population;
        self.st.fitness = new_fitness;
        self.st.constraints = new_constraints;

        for i in 0..pop_size {
            if self.st.constraints[i] && self.st.fitness[i] > self.st.best_f {
                self.st.best_f = self.st.fitness[i];
                self.st.best_x = self.st.pop.row(i).transpose();
            }
        }

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}
