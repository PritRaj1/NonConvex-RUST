# Parallel Tempering

Parallel Tempering with either a Metropolis-Hastings update mechanism or Metropolis-Adjusted Langevin Algorithm (MALA) when gradients are available.

Temperatures are scheduled with a dynamic power law relationship:

```math
t_k = \left( \frac{k}{N_t} \right)^p, \quad k = 0, 1, \ldots, N_t, \quad t_k \in [0, 1], \quad p > 0
```

where $N_t$ is the number of temperatures and $p$ is a parameter that controls the rate of temperature increase. 

In early iterations, $p$ is set large (>1) to cluster more temperatures towards smoother replicas. As iterations progress, $p$ is decreased to <1 to shift more temperatures towards detailed replicas.

This ensures that early iterations are more explorative and later iterations are more exploitative.

## Sources and more information

- [Parallel Tempering](https://arxiv.org/abs/physics/0508111)
- [Power-law scheduling and minimizing KL divergences between temperature distributions](https://doi.org/10.1016/j.csda.2009.07.025)
- [Chains might be initialized between 0 and 1 similar to simulated annealing](https://doi.org/10.13182/NT90-A34350)
- [Metropolis-Hastings](http://www.jstor.org/stable/2280232)
- [MALA](https://doi.org/10.1063/1.436415)
- [Metropolis-Hastings step size adaptation](https://doi.org/10.1007/BF00143556)