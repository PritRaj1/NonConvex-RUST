# MSPO

Multi-Swarm Particle Optimization (MSPO) is a population-based optimization algorithm that uses multiple swarms to explore the search space. 

## Config example

Fully-defined:

```json
{   
    "alg_conf": {
        "MSPO": {
            "num_swarms": 10,
            "swarm_size": 10,
            "w": 0.729,
            "c1": 1.5,
            "c2": 1.5,
            "x_min": 0.0,
            "x_max": 10.0,
            "exchange_interval": 20,
            "exchange_ratio": 0.05
        }
    }
}
```

Default values:

```json
{
    "alg_conf": {
        "MSPO": {}
    }
}
```

## Sources and more information

