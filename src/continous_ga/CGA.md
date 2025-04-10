# Continous Genetic Algorithm

Genetic algorithm for real valued (FP) vectors.

## Selection

| Selection Methods         | Notes                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| Roulette Wheel | - Maintains diversity well                                                                    |
|                          | - Can be biased and not my go-to for constrained problems                                     |
| Tournament     | - More reflective of biological selection and works well for constrained problems             |
|                          | - Weaker individuals can be considered and tournament size must be tuned to maintain diversity|
| Residual (SRS)       | - My preferred method in statistical sampling, has good theoretical properties and maintains diversity |
|                          | - Somewhat deterministic and fast                                                                      |

## Crossover

| Crossover Methods         | Notes                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| Random | - Standard and boring |
| Heuristic     | - More exploitative and faster to converge (with good initial population) |
|      | - More suitable to continous problems (by intuition). Uses a blend of parent characteristics. |

## Sources and more information

- [Continuous Genetic Algorithm](https://doi.org/10.1002/0471671746.ch3)
- [CGA is more akin to Evolutionary Strategies](https://arxiv.org/abs/1703.03864)
- [However, it is still a GA](https://doi.org/10.1007/BFb0029787)
- [Heuristic or blend crossover](https://doi.org/10.1007/978-3-662-03315-9)