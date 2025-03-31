use non_convex_opt::utils::config::{CGAConf};
use non_convex_opt::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction, BooleanConstraintFunction, OptProb};
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

pub struct Rosenbrock<T: FloatNum> {
    pub a: T,
    pub b: T,
}

impl<T: FloatNum> Rosenbrock<T> {
    pub fn new(a: T, b: T) -> Self {
        Self { a, b }
    }
}

impl<T: FloatNum> ObjectiveFunction<T> for Rosenbrock<T> {
    fn f(&self, x: &DVector<T>) -> T {
        let n = x.len();
        let mut sum = T::zero();
        for i in 0..n-1 {
            sum += self.b * (x[i+1] - x[i].powi(2)).powi(2) + 
                   (self.a - x[i]).powi(2);
        }
        sum
    }
}

impl<T: FloatNum> BooleanConstraintFunction<T> for Rosenbrock<T> {
    fn g(&self, x: &DVector<T>) -> DVector<bool> {
        DVector::from_vec(vec![true; x.len()])
    }
}

impl<T: FloatNum> OptProb<T> for Rosenbrock<T> {
    fn objective(&self, x: &DVector<T>) -> T {
        self.f(x)
    }       

    fn constraints(&self, x: &DVector<T>) -> DVector<bool> {
        self.g(x)
    }
}

#[test]
fn test_cga() {
    let conf = CGAConf {
        pop_size: 50,
        num_parents: 10,
        selection_method: "RouletteWheel".to_string(),
        crossover_method: "Random".to_string(),
        crossover_prob: 0.8,
        tournament_size: 2,
    };

    // Initialize population
    let mut init_pop = DMatrix::zeros(conf.pop_size, 2);
    for i in 0..conf.pop_size {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let opt_prob = Rosenbrock::new(1.0, 100.0);
    let mut cga = CGA::new(conf, init_pop, opt_prob);

    // Run a few iterations
    for _ in 0..10 {
        cga.step();
    }

    // Check that we found a reasonable solution
    assert!(cga.best_fitness < 1.0);
}