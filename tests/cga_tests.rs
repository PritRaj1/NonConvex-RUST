use non_convex_opt::utils::config::{CGAConf};
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction, OptProb};
use non_convex_opt::continous_ga::selection::{RouletteWheel, Tournament, Residual};
use non_convex_opt::continous_ga::crossover::{Random, Heuristic};
use non_convex_opt::continous_ga::cga::CGA;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct RosenbrockObjective {
    pub a: f64,
    pub b: f64,
}

impl ObjectiveFunction<f64> for RosenbrockObjective {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..n-1 {
            sum += self.b * (x[i+1] - x[i].powi(2)).powi(2) + 
                   (self.a - x[i]).powi(2);
        }
        sum
    }
}

#[derive(Debug, Clone)]
pub struct RosenbrockConstraints {
    pub a: f64,
    pub b: f64,
}

impl BooleanConstraintFunction<f64> for RosenbrockConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        // Check if all components are within bounds
        x.iter().all(|&xi| xi >= 0.0 && xi <= 1.0)
    }
}

#[test]
fn test_roulette_wheel_selection() {
    let selection = RouletteWheel::new(10, 5);
    let population = DMatrix::<f64>::from_vec(10, 5, vec![1.0; 50]);
    let fitness = DVector::<f64>::from_vec(vec![1.0; 10]);
    let constraint = DVector::from_vec(vec![true; 10]);
    let selected = selection.select(&population, &fitness, &constraint);
    assert_eq!(selected.nrows(), 5);
    assert_eq!(selected.ncols(), 5);
}

#[test]
fn test_tournament_selection() {
    let selection = Tournament::new(10, 5, 2);
    let population = DMatrix::<f64>::from_vec(10, 5, vec![1.0; 50]);
    let fitness = DVector::<f64>::from_vec(vec![1.0; 10]);
    let constraint = DVector::from_vec(vec![true; 10]);
    let selected = selection.select(&population, &fitness, &constraint);
    assert_eq!(selected.nrows(), 5);
    assert_eq!(selected.ncols(), 5);
}

#[test]
fn test_residual_selection() {
    let selection = Residual::new(10, 5);
    let population = DMatrix::<f64>::from_vec(10, 5, vec![1.0; 50]);
    let fitness = DVector::<f64>::from_vec(vec![1.0; 10]);
    let constraint = DVector::from_vec(vec![true; 10]);
    let selected = selection.select(&population, &fitness, &constraint);
    assert_eq!(selected.nrows(), 5);
    assert_eq!(selected.ncols(), 5);
}

#[test]
fn test_random_crossover() {
    let selection = RouletteWheel::new(10, 5);
    let population = DMatrix::<f64>::from_vec(10, 5, vec![1.0; 50]);
    let fitness = DVector::<f64>::from_vec(vec![1.0; 10]);
    let constraint = DVector::from_vec(vec![true; 10]);
    let selected = selection.select(&population, &fitness, &constraint);
    let crossover = Random::new(0.9, 10);
    let offspring = crossover.crossover(&selected);
    assert_eq!(offspring.nrows(), 10);
    assert_eq!(offspring.ncols(), 5);
}

#[test]
fn test_heuristic_crossover() {
    let selection = RouletteWheel::new(10, 5);
    let population = DMatrix::<f64>::from_vec(10, 5, vec![1.0; 50]);
    let fitness = DVector::<f64>::from_vec(vec![1.0; 10]);
    let constraint = DVector::from_vec(vec![true; 10]);
    let selected = selection.select(&population, &fitness, &constraint);
    let crossover = Heuristic::new(0.9, 10);
    let offspring = crossover.crossover(&selected, &fitness);
    assert_eq!(offspring.nrows(), 10);
    assert_eq!(offspring.ncols(), 5);
}

#[test]
fn test_cga() {
    let conf = CGAConf {
        population_size: 50,
        num_parents: 10,
        selection_method: "RouletteWheel".to_string(),
        crossover_method: "Random".to_string(),
        crossover_prob: 0.8,
        tournament_size: 2,
    };

    // Initialize population
    let mut init_pop = DMatrix::zeros(conf.population_size, 2);
    for i in 0..conf.population_size {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let obj_f = RosenbrockObjective{a: 1.0, b: 100.0};
    let constraints = RosenbrockConstraints{a: 1.0, b: 100.0};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    let mut cga = CGA::new(conf, init_pop, opt_prob);

    // Run a few iterations
    for _ in 0..5 {
        cga.step();
    }

    assert!(cga.best_fitness.is_finite());
}