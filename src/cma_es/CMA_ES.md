# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

CMA-ES is a stochastic, derivative-free optimization algorithm for difficult non-linear non-convex optimization problems in continuous domain. It is particularly well suited for ill-conditioned and non-separable problems.

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "CMAES": {
            "population_size": 100,
            "num_parents": 50,
            "initial_sigma": 1.5
        }
    }
}
```

Default values:

```json
{
    "alg_conf": {
        "CMAES": {}
    }
}
```

## Sources and more information

- [CMA-ES + great bibliography](https://cma-es.github.io/)
- [Power iteration eigen decomposition](https://en.wikipedia.org/wiki/Power_iteration)
