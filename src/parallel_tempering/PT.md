# Parallel Tempering

Parallel Tempering with either a Metropolis-Hastings update mechanism or Metropolis-Adjusted Langevin Algorithm (MALA) when gradients are available.

Temperatures are scheduled with a dynamic power law relationship:

```math
t_k = \left( \frac{k}{N_t} \right)^p, \quad k = 0, 1, \ldots, N_t, \quad t_k \in [0, 1], \quad p > 0
```

where $N_t$ is the number of temperatures and $p$ is a parameter that controls the rate of temperature increase. 

In early iterations, $p$ is set large (>1) to cluster more temperatures towards smoother replicas. As iterations progress, $p$ is decreased to <1 to shift more temperatures towards detailed replicas.

This ensures that early iterations are more explorative and later iterations are more exploitative.
