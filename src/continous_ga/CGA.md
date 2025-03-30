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
|      | - More suitable to continous problems (by intuition). Uses a Random of parent characteristics. |