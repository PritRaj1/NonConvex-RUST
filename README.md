# NonConvex-RUST
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Continuous non-convex optimizers implemented in rust for constrained and unconstrained maximization problems. These algorithms were implemented as a side project, but they may be useful and have been open-sourced.

Sources/links to more information in the respective algorithm .md files.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
nonconvex-opt = "0.1.0"
```

## Importing

```rust
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::Config;
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};
use nalgebra::{DVector, DMatrix}; // Works with dynamic
use nalgebra::{SVector, SMatrix}; // Works with static
```

The library works with both statically-sized and dynamically-size vectors. For dynamic examples, see [de_tests](./tests/de_tests.rs), [mspo_tests](./tests/mspo_tests.rs), [nm_tests](./tests/nm_tests.rs), [pt_tests](./tests/pt_tests.rs), or [solver_tests](./tests/solver_tests.rs). For static examples, see any other test.

## Usage

```rust
// Load config from file
let config = Config::new(include_str!("config.json")).unwrap();

// Or create config from JSON string
let config_json = r#"{
    "opt_conf": {
        "max_iter": 1000,
        "rtol": "1e-6", 
        "atol": "1e-6",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "CGA": {
            "common": {
                "num_parents": 2,
            },
            "crossover": {
                "Heuristic": {
                    "crossover_prob": 0.8
                }
            },
            "selection": {
                "Tournament": {
                    "tournament_size": 2
                }
            },
            "mutation": {
                "NonUniform": {
                    "mutation_rate": 0.23,
                    "b": 5.0
                }
            }
        }
    }
}"#;

let config = Config::new(config_json).unwrap();

let mut opt = NonConvexOpt::new(
    config,
    init_x, // Initial population - must be from nalgebra
    obj_f,  // Objective function
    Some(constraints) // Optional constraints
);

// Unconstrained optimization
let mut opt = NonConvexOpt::new(
    config,
    init_x,
    obj_f,
    None::<EmptyConstraints>
);

let result = opt.run();
```
To see the differences between setting up unconstrained and constrained problems, please refer to the [benches/](./benches) subdirectory. See the [examples/](./examples) subdirectory for more direction on using the lib.


## Algorithms

The following GIFs are based on the [2D unconstrained maximization problems](./examples/test_functions.md) in the [examples/](./examples) subdirectory.

### Population-Based 
|  |  |
|-----------|---------------|
| [Continuous Genetic Algorithm (CGA)](./src/algorithms/continous_ga/CGA.md) - Population-based natural selection | <img src="./examples/gifs/cga_kbf.gif" width="300" alt="CGA Example"> |
| <img src="./examples/gifs/pt_kbf.gif" width="300" alt="PT Example"> | [Parallel Tempering (PT)](./src/algorithms/parallel_tempering/PT.md) - Multi-temperature Metropolis-Hastings |
| [Multi-Swarm Particle Optimization (MSPO)](./src/algorithms/multi_swarm/MSPO.md) - Multi-swarm particle optimization | <img src="./examples/gifs/mspo_kbf.gif" width="300" alt="MSPO Example"> |
| <img src="./examples/gifs/de_kbf.gif" width="300" alt="DE Example"> | [Differential Evolution (DE)](./src/algorithms/differential_evolution/DE.md) - Differential evolution |
| [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](./src/algorithms/cma_es/CMA_ES.md) - I love this algorithm! | <img src="./examples/gifs/cmaes_kbf.gif" width="300" alt="CMAES Example"> |


### Local Search 
|  |  |
|-----------|---------------|
| [Tabu Search (TS)](./src/algorithms/tabu_search/TS.md) - Local search with memory | <img src="./examples/gifs/tabu_kbf.gif" width="300" alt="Tabu Example"> |
| <img src="./examples/gifs/grasp_kbf.gif" width="300" alt="GRASP Example"> | [Greedy Randomized Adaptive Search Procedure (GRASP)](./src/algorithms/grasp/GRASP.md) - Construction and local search |

### Gradient-Based 

These work better with mini-batches, and best for unconstrained problems!

|  |  |
|-----------|---------------|
| [Adam](./src/algorithms/adam/ADAM.md) - Adaptive Moment Estimation | <img src="./examples/gifs/adam_kbf.gif" width="300" alt="Adam Example"> |
| <img src="./examples/gifs/sga_kbf.gif" width="300" alt="SGA Example"> | [Stochastic Gradient Ascent (SGA)](./src/algorithms/sg_ascent/SGA.md) - Gradient-based optimization |
| [Limited Memory BFGS (L-BFGS)](./src/algorithms/limited_memory_bfgs/LBFGS.md) - Quasi-Newton gradient-based optimization | <img src="./examples/gifs/lbfgs_kbf.gif" width="300" alt="LBFGS Example"> |

### Direct Search and stochastic optimization

|  |  |
|-----------|---------------|
| [Nelder-Mead](./src/algorithms/nelder_mead/NM.md) - Direct search with simplex | <img src="./examples/gifs/nm_kbf.gif" width="300" alt="Nelder-Mead Example"> |
| <img src="./examples/gifs/sa_kbf.gif" width="300" alt="SA Example"> | [Simulated Annealing (SA)](./src/algorithms/simulated_annealing/SA.md) - Stochastic optimization |

## Config

The config is structured hierarchically, as follows:

- `OptConf` - Optimization configuration
- `AlgConf` - Algorithm configuration, containing one of:
    - `CGAConf` - Continuous Genetic Algorithm configuration
        - `CommonConf` - Common configuration
        - `SelectionConf` - Selection method configuration
        - `CrossoverConf` - Crossover method configuration
        - `MutationConf` - Mutation method configuration
    - `PTConf` - Parallel Tempering configuration
        - `CommonConf` - Common configuration
        - `SwapConf` - Swap configuration
    - `TabuConf` - Tabu Search configuration
        - `CommonConf` - Common configuration
        - `ListType` - List type configuration
            - `StandardConf` - Standard list configuration
            - `ReactiveConf` - Reactive list configuration
    - `GRASPConf` - Greedy Randomized Adaptive Search Procedure configuration
    - `AdamConf` - Adam configuration
    - `SGAConf` - Stochastic Gradient Ascent configuration
    - `NelderMeadConf` - Nelder-Mead configuration
    - `LBFGSConf` - Limited Memory BFGS configuration
        - `CommonConf` - Common configuration
        - `LineSearchConf` - Line search configuration
            - `BacktrackingConf` - Backtracking configuration
            - `StrongWolfeConf` - Strong Wolfe configuration
            - `HagerZhangConf` - Hager Zhang configuration
            - `MoreThuenteConf` - More Thuente configuration
            - `GoldenSectionConf` - Golden Section configuration
    - `MSPOConf` - Multi-Swarm Particle Optimization configuration
    - `SAConf` - Simulated Annealing configuration
    - `DEConf` - Differential Evolution configuration
        - `CommonConf` - Common configuration
        - `MutationType` - Mutation type configuration
            - `StandardConf` - Standard mutation configuration
            - `AdaptiveConf` - Adaptive mutation configuration
    - `CMAESConf` - Covariance Matrix Adaptation Evolution 

Example configs are provided in [tests/jsons/](tests/jsons). More information on each config can be found in the respective algorithm .md files, (links above).

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b my-feature`
3. Make your changes
4. Run tests: `cargo test`
5. Run benchmarks: `cargo bench`    
    - To view the results, run:
    ```bash
    open target/criterion/report/index.html  # on macOS
    xdg-open target/criterion/report/index.html  # on Linux
    start target/criterion/report/index.html  # on Windows
    ```
6. Add sources and more information to the respective algorithm .md file - so that others can learn and share too!
7. Commit and push
8. Open a Pull Request

## License

This project is open-sourced under the [MIT License](LICENSE).

## TODO

| Algorithm | Status |
|-----------|---------------|
| CGA | [√] |
| PT | [√] |
| Tabu | [√] |
| GRASP | [√] |
| Adam | [√] |
| SGA | [√] |
| Nelder-Mead | [√] |
| LBFGS | [√] |
| MSPO | [√] |
| SA | [√] |
| DE | [√] |
| CMAES | [√] |