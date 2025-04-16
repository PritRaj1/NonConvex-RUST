# Tabu Search

Tabu search is a local search algorithm that uses a tabu list to prevent the search from revisiting the same solution, thereby escaping local optima.

Two variants of tabu search are implemented:

- Standard tabu search
    - Uses a fixed size tabu list.
- Reactive tabu search
    - Uses a dynamic tabu list size.

## Guidance

|  | Standard Tabu Search | Reactive Tabu Search |
|---------|---------------------|---------------------|
| Tabu List Size | Fixed size specified by parameter | Dynamically adjusts between min and max size |
| Memory Usage | Constant memory usage | Variable memory usage based on search progress |
| Parameter Tuning | Requires careful tuning of tabu list size | More robust to initial parameter settings |
| Adaptation | No adaptation during search | Adapts tabu list size based on search effectiveness |
| Escape Mechanism | Basic tabu restrictions | Enhanced escape from local optima through size adjustments |
| When to Use | Well-understood problem spaces | Problems with varying landscape complexity |


## Sources and more information

- [Tabu Search](https://ieeexplore.ieee.org/document/9091743)
- [Reactive Tabu Search](https://doi.org/10.1287/ijoc.6.2.126)