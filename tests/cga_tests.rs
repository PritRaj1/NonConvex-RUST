mod common;
use non_convex_opt::utils::config::CGAConf;
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{RosenbrockObjective, RosenbrockConstraints};
use non_convex_opt::continous_ga::selection::{RouletteWheel, Tournament, Residual};
use non_convex_opt::continous_ga::crossover::{Random, Heuristic};
use non_convex_opt::continous_ga::cga::CGA;
use nalgebra::{DMatrix, DVector};

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

    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    let mut cga = CGA::new(conf, init_pop, opt_prob);

    // Run a few iterations
    for _ in 0..5 {
        cga.step();
    }

    assert!(cga.best_fitness.is_finite());
}