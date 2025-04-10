use non_convex_opt::{NonConvexOpt};
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, CGAConf, PTConf};
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};
use nalgebra::{DVector, DMatrix};

#[derive(Clone)]
struct Rosenbrock;

impl ObjectiveFunction<f64> for Rosenbrock {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..(x.len() - 1) {
            let xi = x[i];
            let xi1 = x[i + 1];
            sum += 100.0 * (xi1 - xi * xi).powi(2) + (1.0 - xi).powi(2);
        }
        sum
    }
}

#[derive(Clone)]
struct BoxConstraints {
    lower: f64,
    upper: f64,
}

impl BooleanConstraintFunction<f64> for BoxConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        x.iter().all(|&xi| xi >= self.lower && xi <= self.upper)
    }
}

#[test]
fn test_cga() {
    let conf = Config {
        opt_conf: OptConf {
            max_iter: 1000,
            rtol: 1e-2,
            atol: 1e-2,
        },
        alg_conf: AlgConf::CGA(CGAConf {
            population_size: 50,
            num_parents: 10,
            selection_method: "RouletteWheel".to_string(),
            crossover_method: "Random".to_string(),
            crossover_prob: 0.8,
            tournament_size: 2,
        }),
    };

    let mut init_pop = DMatrix::zeros(50, 2);
    for i in 0..50 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let constraints = BoxConstraints { lower: -2.0, upper: 2.0 };
    let mut opt = NonConvexOpt::new(conf, init_pop.clone(), Rosenbrock, Some(constraints));

    let initial_best_fitness: f64 = init_pop.row_iter()
        .map(|row| Rosenbrock.f(&row.transpose()))
        .fold(f64::INFINITY, |a, b| a.min(b));

    let result = opt.run();

    println!("Initial best fitness: {}", initial_best_fitness);
    println!("Best f: {}", result.best_f);

    assert!(-result.best_f.exp() < 0.01);
    assert!(result.best_f > initial_best_fitness);
}

#[test]
fn test_pt() {
    let conf = Config {
        opt_conf: OptConf {
            max_iter: 2,
            rtol: 1e-2,
            atol: 1e-2,
        },
        alg_conf: AlgConf::PT(PTConf {
            num_replicas: 1000,
            num_chains: 10,
            power_law_init: 2.0,
            power_law_final: 0.5,
            power_law_cycles: 1,
            alpha: 0.1,
            omega: 2.1,
            swap_check_type: "Always".to_string(),
            swap_frequency: 1.0,
            swap_probability: 0.1,
            mala_step_size: 0.01,
        }),
    };

    let mut init_pop = DMatrix::zeros(10, 2);
    for i in 0..10 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let constraints = BoxConstraints { lower: -2.0, upper: 2.0 };
    let mut opt = NonConvexOpt::new(conf, init_pop.clone(), Rosenbrock, Some(constraints));

    let initial_best_fitness: f64 = init_pop.row_iter()
        .map(|row| Rosenbrock.f(&row.transpose()))
        .fold(f64::INFINITY, |a, b| a.min(b));

    println!("Initial best fitness: {}", initial_best_fitness);
    
    let result = opt.run();

    println!("Best f: {}", result.best_f);

    assert!(-result.best_f.exp() < 0.01);
    assert!(result.best_f > initial_best_fitness);
}   