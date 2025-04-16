# Greedy Randomized Adaptive Search Procedures

GRASP (Greedy Randomized Adaptive Search Procedures) is a multi-start metaheuristic algorithm for combinatorial optimization problems. Samples randomly, choooses greedily!

## Construction Phase
The construction phase builds solutions incrementally by randomly selecting elements from the RCL - a subset of best candidates determined by a threshold parameter α. This provides controlled randomization while maintaining solution quality.

## Local Search Phase 
The local search phase improves constructed solutions by iteratively exploring neighboring solutions until reaching a local optimum. This intensification step helps refine promising solutions found during construction.


## Sources and further information

- [GRASP](https://link.springer.com/chapter/10.1007/0-306-48056-5_8)