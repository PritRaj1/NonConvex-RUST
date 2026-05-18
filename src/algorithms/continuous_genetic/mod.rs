mod algo;
mod config;
mod crossover;
mod mutation;
mod selection;

pub use algo::CGA;
pub use config::*;
pub use crossover::{CrossoverOperator, Heuristic, Random, SimulatedBinary};
pub use mutation::{
    Gaussian, MutationOperator, MutationOperatorEnum, NonUniform, Polynomial, Uniform,
};
pub use selection::{Residual, RouletteWheel, SelectionOperator, Tournament};
