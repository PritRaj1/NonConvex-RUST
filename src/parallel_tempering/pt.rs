use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::parallel_tempering::replica_exchange::{SwapCheck, Periodic, Stochastic, Always};
use crate::parallel_tempering::metropolis_hastings::MetropolisHastings;
use crate::utils::config::PTConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

pub struct PT<T: FloatNum, F: OptProb<T>> {
    pub conf: PTConf,
    pub metropolis_hastings: MetropolisHastings<T, F>,
    pub swap_check: SwapCheck,
    pub p_schedule: Vec<T>,
    pub population: Vec<DMatrix<T>>, // For each replica
    pub fitness: Vec<DVector<T>>,
    pub constraints: Vec<DVector<bool>>,
    pub opt_prob: F,
    pub best_individual: DVector<T>,
    pub best_fitness: T,
}

impl<T: FloatNum, F: OptProb<T>> PT<T, F> {
    pub fn new(conf: PTConf, init_pop: DMatrix<T>, opt_prob: F) -> Self {

        let swap_check = match conf.swap_check.as_str() {
            "Periodic" => SwapCheck::Periodic(Periodic::new(conf.swap_frequency, conf.total_steps)),
            "Stochastic" => SwapCheck::Stochastic(Stochastic::new(conf.swap_probability)),
            "Always" => SwapCheck::Always(Always::new()),
            _ => panic!("Invalid swap check"),
        };

        // Power law schedule for cyclic annealing
        let p_init = conf.power_law_init;
        let p_final = conf.power_law_final;
        let p_cycles = conf.power_law_cycles;
        let p_schedule = (0..p_cycles).map(|i| p_init * (p_final / p_init) * (i as f64 / p_cycles as f64).powf(p_init)).collect();
        let num_iters = conf.num_iters;

        let x: Vec<T> = (0..=num_iters)
            .map(|i| T::from_f64(2.0 * std::f64::consts::PI * (p_cycles as f64 + 0.5) * (i as f64 / num_iters as f64)).unwrap())
            .collect();

        let p_schedule: Vec<T> = x.iter()
            .map(|&xi| p_init + (p_final - p_init) * T::from_f64(0.5).unwrap() * (T::from_f64(1.0).unwrap() - xi.cos()))
            .collect();

        // Calculate initial fitness and constraints in parallel for all replicas
        let (fitness, constraints): (Vec<DVector<T>>, Vec<DVector<bool>>) = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = init_pop.row(i).transpose();
                let fit = opt_prob.objective(&individual);
                let constr = opt_prob.constraints(&individual)[0];
                (fit, constr)
            })
            .unzip();

        let fitness = DVector::from_vec(fitness);
        let constraints = DVector::from_vec(constraints);

        Self { swap_check, p_schedule, metropolis_hastings, population, fitness, constraints, opt_prob, best_individual, best_fitness }
    }
    let p_cycles = conf.power_law_cycles;
    let p_schedule = (0..p_cycles).map(|i| p_init * (p_final / p_init) * (i as f64 / p_cycles as f64).powf(p_init)).collect();
    let num_iters = conf.num_iters;

    let x: Vec<T> = (0..=num_iters)
        .map(|i| T::from_f64(2.0 * std::f64::consts::PI * (p_cycles as f64 + 0.5) * (i as f64 / num_iters as f64)).unwrap())
        .collect();

    let p_schedule: Vec<T> = x.iter()
        .map(|&xi| p_init + (p_final - p_init) * T::from_f64(0.5).unwrap() * (T::from_f64(1.0).unwrap() - xi.cos()))
        .collect();

    // Calculate initial fitness and constraints in parallel for all replicas

    pub fn swap(&mut self) {
}