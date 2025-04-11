# NonConvex-RUST
Non-convex optimizers implemented in RUST for constrained and unconstrained maximization problems. 

Sources/links to more information in the respective algorithm .md files.

## Examples

The following GIFs are based on the [2D unconstrained maximization problem](./examples/test_function.md) in the 'examples' subdirectory.

| Algorithm | Description | Visualization |
|-----------|-------------|---------------|
| [Continuous Genetic Algorithm (CGA)](./src/continous_ga/CGA.md) | Population-based natural selection | <img src="./examples/cga_kbf.gif" width="300" alt="CGA Example"> |
| [Parallel Tempering (PT)](./src/parallel_tempering/PT.md) | Multi-temperature Monte Carlo sampling | <img src="./examples/pt_kbf.gif" width="300" alt="PT Example"> |

