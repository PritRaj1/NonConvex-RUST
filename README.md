# NonConvex-RUST
Non-convex optimizers implemented in RUST for constrained and unconstrained maximization problems. 

Sources/links to more information in the respective algorithm .md files.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
non_convex_opt = "0.1.0"
```

## Importing

```rust
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, CGAConf, PTConf, TabuConf};
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};
use nalgebra::{DVector, DMatrix};
```

## Examples

The following GIFs are based on the [2D unconstrained maximization problem](./examples/test_function.md) in the 'examples' subdirectory.

|  |  |
|-----------|---------------|
| [Continuous Genetic Algorithm (CGA)](./src/continous_ga/CGA.md) - Population-based natural selection | <img src="./examples/cga_kbf.gif" width="300" alt="CGA Example"> |
| <img src="./examples/pt_kbf.gif" width="300" alt="PT Example"> | [Parallel Tempering (PT)](./src/parallel_tempering/PT.md) - Multi-temperature Monte Carlo sampling |
| [Tabu Search (TS)](./src/tabu_search/tabu.md) - Local search with memory | <img src="./examples/tabu_kbf.gif" width="300" alt="Tabu Example"> |

## Usage

```rust
// Load config from file
let config = Config::new(include_str!("config.json")).unwrap();

// Or create config directly
let config = Config {
    opt_conf: OptConf {
        max_iter: 1000,
        rtol: 1e-6,
        atol: 1e-6,
    },
    alg_conf: AlgConf::CGA(CGAConf {
        population_size: 100,
        num_parents: 2,
        selection_method: "Tournament".to_string(),
        crossover_method: "Random".to_string(),
        crossover_prob: 0.8,
        tournament_size: 2,
    }),
};

let mut opt = NonConvexOpt::new(
    config,
    init_x, // Initial population - must be a DMatrix from nalgebra
    obj_f,  // Objective function
    Some(constraints) // Optional constraints
);

let result = opt.run();
```
## Config

The config is structured as follows:

- `OptConf` - Optimization configuration
- `AlgConf` - Algorithm configuration
- `CGAConf` - Continuous Genetic Algorithm configuration
- `PTConf` - Parallel Tempering configuration
- `TabuConf` - Tabu Search configuration

The default values are:

```json
    {
        "opt_conf": {
            "max_iter": 1000,
            "rtol": 1e-6,
            "atol": 1e-6
        },
    "alg_conf": {
        "cga": {
            "population_size": 100,
            "num_parents": 2,
            "selection_method": "tournament",
            "crossover_method": "uniform",
            "crossover_prob": 0.8,
            "tournament_size": 2
        },
        "pt": {
            "num_replicas": 10,
            "power_law_init": 2.0,
            "power_law_final": 0.5,
            "power_law_cycles": 1,
            "alpha": 0.1,
            "omega": 2.1,
            "swap_check_type": "Always",
            "swap_frequency": 1.0,
            "swap_probability": 0.1,
            "mala_step_size": 0.01
        },
        "ts": {
            "tabu_list_size": 20,
            "num_neighbors": 50,
            "step_size": 0.1,
            "perturbation_prob": 0.3,
            "tabu_threshold": 1e-6
        }
    }
}
```

See the 'examples' subdirectory for more information on how to the lib.