use non_convex_opt::{NonConvexOpt, Result};
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, CGAConf};
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};
use nalgebra::{DVector, DMatrix};

#[derive(Clone)]
struct Sphere;

impl ObjectiveFunction<f64> for Sphere {
    fn f(&self, x: &DVector<f64>) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
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
fn test_cga_sphere() {
    let conf = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 1e-6,
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
    let mut opt = NonConvexOpt::new(conf, init_pop, Sphere, Some(constraints));
    let result = opt.run();

    assert!(result.best_f < 1.0);
}