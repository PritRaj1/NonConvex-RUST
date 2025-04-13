use non_convex_opt::utils::config::TabuConf;
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction, OptProb};
use non_convex_opt::tabu_search::tabu::TabuSearch;
use nalgebra::DVector;

#[derive(Clone)]
struct RosenbrockObjective;

impl ObjectiveFunction<f64> for RosenbrockObjective {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..n-1 {
            sum += 100.0 * (x[i+1] - x[i].powi(2)).powi(2) + 
                   (1.0 - x[i]).powi(2);
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
fn test_tabu_search() {
    let conf = TabuConf {
        tabu_list_size: 20,
        num_neighbors: 50,
        step_size: 0.1,
        perturbation_prob: 0.3,
        tabu_threshold: 1e-6,
    };

    let init_x = DVector::from_vec(vec![0.0, 0.0]);
    let obj_f = RosenbrockObjective;
    let constraints = BoxConstraints { lower: -2.0, upper: 2.0 };
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut tabu = TabuSearch::new(conf, init_x.clone(), opt_prob);
    
    let initial_fitness = tabu.best_fitness;
    
    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.best_fitness > initial_fitness);
    assert!(tabu.best_x.iter().all(|&x| x.abs() <= 2.0));
} 