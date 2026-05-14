use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::{rngs::StdRng, Rng};
use rayon::prelude::*;
use std::collections::VecDeque;

use super::config::{CoolingScheduleType, RestartStrategy, SAConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};
use crate::utils::rng;

use crate::algorithms::simulated_annealing::{
    acceptance::MetropolisAcceptance,
    cooling::{
        AdaptiveCooling, CauchyCooling, CoolingSchedule, ExponentialCooling, LogarithmicCooling,
    },
    neighbor_gen::GaussianGenerator,
    stagnation_monitor::SAStagnationMonitor,
};

pub struct SimulatedAnnealing<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub conf: SAConf,
    pub opt_prob: OptProb<T, D>,
    pub st: State<T, N, D>,
    pub temperature: T,

    x: OVector<T, D>,
    fitness: T,
    constraints: bool,
    stagnation_monitor: SAStagnationMonitor<T>,
    improvement_history: VecDeque<f64>,
    success_history: VecDeque<bool>,
    restart_counter: usize,
    last_restart_iter: usize,
    current_step_size: T,
    current_cooling_rate: T,
    neighbor_gen: GaussianGenerator<T, D>,
    cooling_schedule: Box<dyn CoolingSchedule<T> + Send + Sync>,
    acceptance: MetropolisAcceptance<T, D>,
    stagnation_window: usize,
    rng: StdRng,
}

impl<T, N, D> SimulatedAnnealing<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(
        conf: SAConf,
        init_pop: OMatrix<T, U1, D>,
        opt_prob: OptProb<T, D>,
        stagnation_window: usize,
        seed: u64,
    ) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let n = init_x.len();
        let improvement_threshold = T::cast(1e-6); // TODO: should this be hard-coded?
        let stagnation_monitor =
            SAStagnationMonitor::new(improvement_threshold, best_f, stagnation_window);

        let cooling_schedule: Box<dyn CoolingSchedule<T> + Send + Sync> =
            match conf.advanced.cooling_schedule {
                CoolingScheduleType::Exponential => Box::new(ExponentialCooling),
                CoolingScheduleType::Logarithmic => Box::new(LogarithmicCooling),
                CoolingScheduleType::Cauchy => Box::new(CauchyCooling),
                CoolingScheduleType::Adaptive => Box::new(AdaptiveCooling),
            };

        Self {
            conf: conf.clone(),
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            fitness: best_f,
            constraints: opt_prob.is_feasible(&init_x),
            st: State {
                best_x: init_x.clone(),
                best_f,
                pop: OMatrix::<T, N, D>::from_fn_generic(
                    N::from_usize(1),
                    D::from_usize(n),
                    |_, j| init_x.clone()[j],
                ),
                fitness: OVector::<T, N>::from_element_generic(N::from_usize(1), U1, best_f),
                constraints: OVector::<bool, N>::from_element_generic(
                    N::from_usize(1),
                    U1,
                    opt_prob.is_feasible(&init_x.clone()),
                ),
                iter: 1,
            },
            temperature: T::cast(conf.initial_temp),
            stagnation_monitor,
            improvement_history: VecDeque::with_capacity(conf.advanced.improvement_history_size),
            success_history: VecDeque::with_capacity(conf.advanced.success_history_size),
            restart_counter: 0,
            last_restart_iter: 1,
            current_step_size: T::cast(conf.step_size),
            current_cooling_rate: T::cast(conf.cooling_rate),
            neighbor_gen: GaussianGenerator::new(
                opt_prob.clone(),
                init_x.clone(),
                T::cast(conf.step_size),
                seed,
            ),
            cooling_schedule,
            acceptance: MetropolisAcceptance::new(opt_prob, init_x, seed),
            stagnation_window,
            rng: rng::seeded(seed),
        }
    }

    fn check_restart(&mut self) -> bool {
        match &self.conf.advanced.restart_strategy {
            RestartStrategy::None => false,
            RestartStrategy::Periodic { frequency } => {
                self.st.iter - self.last_restart_iter >= *frequency
            }
            RestartStrategy::Stagnation {
                max_iterations,
                threshold,
            } => {
                self.stagnation_monitor.stagnation_counter() >= *max_iterations
                    || self
                        .improvement_history
                        .back()
                        .map(|v| *v < *threshold)
                        .unwrap_or(false)
            }
            RestartStrategy::Adaptive {
                base_frequency,
                adaptation_rate,
            } => {
                let adaptive_frequency = (*base_frequency as f64
                    * (1.0 + adaptation_rate * self.stagnation_monitor.stagnation_counter() as f64))
                    as usize;
                self.st.iter - self.last_restart_iter >= adaptive_frequency
            }
            RestartStrategy::Diversity { min_diversity } => {
                if self.improvement_history.len() < 5 {
                    false
                } else {
                    let variance = self.calculate_improvement_variance();
                    variance < *min_diversity
                }
            }
        }
    }

    fn perform_restart(&mut self) {
        let current_best = self.st.best_x.clone();
        let current_best_f = self.st.best_f;

        // Reset to best known solution with some perturbation
        let dim = current_best.len();

        for i in 0..dim {
            let perturbation = T::cast(self.rng.random_range(-0.1..0.1));
            self.x[i] = current_best[i] + perturbation;
        }

        let bounds = (T::cast(self.conf.x_min), T::cast(self.conf.x_max));
        for i in 0..self.x.len() {
            self.x[i] = self.x[i].clamp(bounds.0, bounds.1);
        }

        self.fitness = self.opt_prob.evaluate(&self.x);
        self.constraints = self.opt_prob.is_feasible(&self.x);

        self.stagnation_monitor = SAStagnationMonitor::new(
            T::cast(1e-6), // TODO: should this be hard-coded?
            current_best_f,
            self.stagnation_window,
        );

        self.restart_counter += 1;
        self.last_restart_iter = self.st.iter;
        self.improvement_history.clear();
        self.success_history.clear();

        self.temperature = self
            .cooling_schedule
            .reheat(T::cast(self.conf.initial_temp));

        self.current_step_size = T::cast(self.conf.step_size);
        self.current_cooling_rate = T::cast(self.conf.cooling_rate);
    }

    fn calculate_improvement_variance(&self) -> f64 {
        if self.improvement_history.len() < 2 {
            return 0.0;
        }

        let mean =
            self.improvement_history.iter().sum::<f64>() / self.improvement_history.len() as f64;
        let variance = self
            .improvement_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.improvement_history.len() as f64;

        variance
    }

    fn adapt_parameters(&mut self) {
        if !self.conf.advanced.adaptive_parameters {
            return;
        }

        if self.success_history.len() < 5 {
            return;
        }

        let (temp_factor, _) = self.stagnation_monitor.get_adaptation_suggestions();

        self.temperature *= T::cast(temp_factor);

        let success_rate = self.success_history.iter().filter(|&&x| x).count() as f64
            / self.success_history.len() as f64;

        if success_rate < 0.2 {
            self.current_cooling_rate *= T::cast(0.99);
        } else if success_rate > 0.6 {
            self.current_cooling_rate *= T::cast(1.01);
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for SimulatedAnnealing<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        if self.check_restart() {
            self.perform_restart();
            return;
        }

        let min_step = self.conf.step_size * 0.01;
        let step_size_f64 = (self.current_step_size.to_f64().unwrap()
            * (self.temperature / T::cast(self.conf.initial_temp))
                .to_f64()
                .unwrap()
                .sqrt())
        .max(min_step);

        let bounds = (T::cast(self.conf.x_min), T::cast(self.conf.x_max));

        let neighbors: Vec<_> = (0..self.conf.num_neighbors)
            .into_par_iter()
            .map(|_| {
                let mut local_ng = self.neighbor_gen.clone();
                local_ng.generate(&self.x, step_size_f64, bounds, self.temperature)
            })
            .collect();

        let mut accepted_count = 0;

        for neighbor in neighbors {
            let neighbor_fitness = self.opt_prob.evaluate(&neighbor);

            if neighbor_fitness > self.st.best_f && self.opt_prob.is_feasible(&neighbor) {
                self.st.best_f = neighbor_fitness;
                self.st.best_x = neighbor.clone();
            }

            let feasible = self.opt_prob.is_feasible(&neighbor);

            if self.acceptance.accept(
                &self.x,
                self.fitness,
                &neighbor,
                neighbor_fitness,
                self.temperature,
                T::cast(step_size_f64),
            ) && feasible
            {
                self.x = neighbor;
                self.fitness = neighbor_fitness;
                self.constraints = feasible;
                accepted_count += 1;
            }
        }

        let success_rate = accepted_count as f64 / self.conf.num_neighbors as f64;
        self.stagnation_monitor.check_stagnation(
            self.st.best_f,
            self.temperature,
            T::cast(step_size_f64),
        );

        let improvement = self.st.best_f - self.fitness;
        let improvement_f64 = improvement.to_f64().unwrap_or(0.0);
        self.improvement_history.push_back(improvement_f64);
        if self.improvement_history.len() > self.conf.advanced.improvement_history_size {
            self.improvement_history.pop_front();
        }

        let was_accepted = accepted_count > 0;
        self.success_history.push_back(was_accepted);
        if self.success_history.len() > self.conf.advanced.success_history_size {
            self.success_history.pop_front();
        }

        let success_rate_for_cooling = success_rate;

        if self.conf.use_adaptive_cooling {
            self.temperature = self.cooling_schedule.adaptive_temperature(
                T::cast(self.conf.initial_temp),
                self.st.iter,
                self.current_cooling_rate,
                success_rate_for_cooling,
            );
        } else {
            self.temperature = self.cooling_schedule.temperature(
                T::cast(self.conf.initial_temp),
                self.st.iter,
                self.current_cooling_rate,
            );
        }

        let min_temp = T::cast(self.conf.initial_temp) * T::cast(self.conf.min_temp_factor);
        self.temperature = self.temperature.max(min_temp);

        self.adapt_parameters();

        if self.fitness > self.st.best_f {
            self.st.best_f = self.fitness;
            self.st.best_x = self.x.clone();
        }

        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = self.fitness;
        self.st.constraints[0] = self.constraints;

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}
