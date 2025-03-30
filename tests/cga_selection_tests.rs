use non_convex_opt::continous_ga::selection::{RouletteWheel, Tournament, Residual};
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
