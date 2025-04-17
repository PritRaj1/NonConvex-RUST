# Greedy Randomized Adaptive Search Procedures

GRASP (Greedy Randomized Adaptive Search Procedures) is a multi-start metaheuristic algorithm for combinatorial optimization problems. Samples randomly, choooses greedily!

## Construction Phase
The construction phase builds solutions incrementally by randomly selecting elements from the RCL - a subset of best candidates determined by a threshold parameter α. This provides controlled randomization while maintaining solution quality.

## Local Search Phase 
The local search phase improves constructed solutions by iteratively exploring neighboring solutions until reaching a local optimum. This intensification step helps refine promising solutions found during construction.

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "GRASP": {
            "num_candidates": 100,
            "alpha": 0.1,
            "num_neighbors": 50,
            "step_size": 0.1,
            "perturbation_prob": 0.1
        }
    }
}   
```

Default values, (nothing needs to be specified):

```json
{
    "alg_conf": {
        "GRASP": {}
    }
}
```

## Sources and further information

- [GRASP](https://link.springer.com/chapter/10.1007/0-306-48056-5_8)