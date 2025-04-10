use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::parallel_tempering::replica_exchange::{SwapCheck, Periodic, Stochastic, Always};
use crate::parallel_tempering::metropolis_hastings::{MetropolisHastings, update_step_size};
use crate::utils::config::PTConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};

pub struct PT<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: PTConf,
    pub metropolis_hastings: MetropolisHastings<T, F, G>,
    pub swap_check: SwapCheck,
    pub p_schedule: Vec<T>,
    pub population: Vec<DMatrix<T>>, // For each replica
    pub fitness: Vec<DVector<T>>,
    pub constraints: Vec<DVector<bool>>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_individual: DVector<T>,
    pub best_fitness: T,
    pub iter: usize,
    pub step_sizes: Vec<Vec<DMatrix<T>>>,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> PT<T, F, G> {
    pub fn new(conf: PTConf, init_pop: DMatrix<T>, opt_prob: OptProb<T, F, G>, max_iter: usize) -> Self {

        let swap_check = match conf.swap_check_type.as_str() {
            "Periodic" => SwapCheck::Periodic(Periodic::new(conf.swap_frequency, max_iter)),
            "Stochastic" => SwapCheck::Stochastic(Stochastic::new(conf.swap_probability)),
            "Always" => SwapCheck::Always(Always::new()),
            _ => panic!("Invalid swap check"),
        };

        let metropolis_hastings = MetropolisHastings::new(opt_prob.clone(), T::from_f64(conf.mala_step_size).unwrap());

        // Power law schedule for cyclic annealing
        let p_init = conf.power_law_init;
        let p_final = conf.power_law_final;
        let p_cycles = conf.power_law_cycles;
        let num_iters = max_iter;

        let x: Vec<f64> = (0..=num_iters)
            .map(|i| 2.0 * std::f64::consts::PI * (p_cycles as f64 + 0.5) * (i as f64 / num_iters as f64))
            .collect();

        let p_schedule: Vec<T> = x.iter()
            .map(|&xi| T::from_f64(p_init + (p_final - p_init) * 0.5 * (1.0 - xi.cos())).unwrap())
            .collect();

        // Initialize populations in parallel
        let init_results: Vec<(DMatrix<T>, DVector<T>, DVector<bool>)> = (0..conf.num_replicas)
            .into_par_iter()
            .map(|_| {
                let mut pop = DMatrix::zeros(init_pop.nrows(), init_pop.ncols());
                for i in 0..init_pop.nrows() {
                    pop.set_row(i, &init_pop.row(i));
                }

                let fit: Vec<T> = (0..init_pop.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let individual = init_pop.row(i).transpose();
                        opt_prob.objective.f(&individual)
                    })
                    .collect();

                let constr: Vec<bool> = (0..init_pop.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let individual = init_pop.row(i).transpose();
                        opt_prob.is_feasible(&individual)
                    })
                    .collect();

                (
                    pop,
                    DVector::from_vec(fit),
                    DVector::from_vec(constr)
                )
            })
            .collect();

        // Unzip the results
        let mut population = Vec::with_capacity(conf.num_replicas);
        let mut fitness = Vec::with_capacity(conf.num_replicas);
        let mut constraints = Vec::with_capacity(conf.num_replicas);

        for (pop, fit, constr) in init_results {
            population.push(pop);
            fitness.push(fit);
            constraints.push(constr);
        }

        // Find best individual across all replicas
        let mut best_idx = 0;
        let mut best_fitness = fitness[0][0];
        for i in 0..conf.num_replicas {
            for j in 0..fitness[i].len() {
                if fitness[i][j] < best_fitness && constraints[i][j] {
                    best_fitness = fitness[i][j];
                    best_idx = i;
                }
            }
        }

        let best_individual = population[best_idx].row(0).transpose();
        let iter = 0;
        let step_sizes: Vec<Vec<DMatrix<T>>> = (0..conf.num_replicas)
            .map(|_| {
                (0..population[0].nrows())
                    .map(|_| DMatrix::identity(population[0].ncols(), population[0].ncols()))
                    .collect()
            })
            .collect();

        Self { 
            conf,
            metropolis_hastings,
            swap_check,
            p_schedule,
            population,
            fitness,
            constraints,
            opt_prob,
            best_individual,
            best_fitness,
            iter,
            step_sizes,
        }
    }

    // Replica exchange
    pub fn swap(&mut self) {
        let n = self.population.len() - 1;
        let m = self.population[0].nrows();

        // Initialize swap matrix
        let mut swap_bool = DMatrix::from_element(n, m, false);

        // Determine which pairs to swap
        let swap_results: Vec<Vec<(usize, usize, bool)>> = (0..n).into_par_iter().map(|i| {
            (0..m).into_par_iter().map(|j| {
                let x_old = self.population[i].row(j).transpose();
                let x_new = self.population[i+1].row(j).transpose();
                let constraints_new = self.constraints[i+1][(j, 0)];
                let t = self.p_schedule[self.iter];
                let t_swap = self.p_schedule[self.iter+1];
                let accept = self.metropolis_hastings.accept_reject(&x_old, &x_new, constraints_new, t, t_swap);
                (i, j, accept)
            }).collect()
        }).collect();

        for swap_result in swap_results {
            for (i, j, accept) in swap_result {
                swap_bool[(i, j)] = accept;
            }
        }

        // Perform swaps
        let mut new_population = self.population.clone();
        let mut new_fitness = self.fitness.clone();
        let mut new_constraints = self.constraints.clone();

        for i in 0..n {
            for j in 0..m {
                if swap_bool[(i, j)] {
                    // Swap population rows
                    let temp_row = self.population[i].row(j).clone_owned();
                    new_population[i].set_row(j, &self.population[i+1].row(j));
                    new_population[i+1].set_row(j, &temp_row);

                    // Swap fitness values
                    let temp_fit = self.fitness[i][(j, 0)];
                    new_fitness[i][(j, 0)] = self.fitness[i+1][(j, 0)];
                    new_fitness[i+1][(j, 0)] = temp_fit;

                    // Swap constraints
                    let temp_const = self.constraints[i][(j, 0)];
                    new_constraints[i][(j, 0)] = self.constraints[i+1][(j, 0)];
                    new_constraints[i+1][(j, 0)] = temp_const;
                }
            }
        }

        self.population = new_population;
        self.fitness = new_fitness;
        self.constraints = new_constraints;
    }
            
    pub fn step(&mut self) {
        let temperatures: Vec<T> = (0..self.conf.num_replicas)
            .map(|k| {
                let power = self.p_schedule[self.iter].to_f64().unwrap();
                T::from_f64((k as f64 / self.conf.num_replicas as f64).powf(power)).unwrap()
            })
            .collect();
        
        // Local move
        let updates: Vec<Vec<Option<(DVector<T>, T, bool, DMatrix<T>)>>> = (0..self.conf.num_replicas)
            .into_par_iter()
            .map(|i| {
                (0..self.population[i].nrows())
                    .into_par_iter()
                    .map(|j| {
                        let x_old = self.population[i].row(j).transpose();
                        let x_new = self.metropolis_hastings.local_move(&x_old, &self.step_sizes[i][j]);
                        let constr_new = self.opt_prob.is_feasible(&x_new);
                        
                        if self.metropolis_hastings.accept_reject(
                            &x_old,
                            &x_new,
                            constr_new,
                            temperatures[i],
                            -T::from_f64(1.0).unwrap() // Send in negative to signal local move 
                        ) {
                            let new_step_size = if self.opt_prob.objective.gradient(&x_old).is_none() {
                                update_step_size(
                                    &self.step_sizes[i][j],
                                    &x_old,
                                    &x_new,
                                    T::from_f64(self.conf.alpha).unwrap(),
                                    T::from_f64(self.conf.omega).unwrap()
                                )
                            } else {
                                self.step_sizes[i][j].clone()
                            };
                            
                            let fitness_new = self.opt_prob.objective.f(&x_new);
                            
                            Some((
                                x_new.clone(),
                                fitness_new,
                                constr_new,
                                new_step_size
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // Apply updates
        for (i, replica) in updates.iter().enumerate() {
            for (j, update) in replica.iter().enumerate() {
                if let Some((x_new, fitness_new, constr_new, step_size_new)) = update {
                    self.population[i].set_row(j, &x_new.transpose());
                    self.fitness[i][j] = *fitness_new;
                    self.constraints[i][j] = *constr_new;
                    self.step_sizes[i][j] = step_size_new.clone();
                }
            }
        }

        // Replica exchange
        if match &self.swap_check {
            SwapCheck::Periodic(p) => p.should_swap(self.iter),
            SwapCheck::Stochastic(s) => s.should_swap(self.iter),
            SwapCheck::Always(a) => a.should_swap(self.iter),
        } {
            self.swap();
        }

        // Update best individual
        let mut best_idx = 0;
        let mut best_fitness = self.fitness[0][0];
        for i in 0..self.conf.num_replicas {
            for j in 0..self.fitness[i].len() {
                if self.fitness[i][j] < best_fitness && self.constraints[i][j] {
                    best_fitness = self.fitness[i][j];
                    best_idx = i;
                }
            }
        }
        self.best_individual = self.population[best_idx].row(0).transpose();
        self.best_fitness = best_fitness;

        self.iter += 1;
    }
}