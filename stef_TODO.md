# TODO

## Programming
* Control output of optimization:
    - Loss function AND 
    - Objective function (= Loss + Regularization)
* Implement bfgs for both, theta as well as the classifier!

* If batch != examples: Initialize u->Ytrain with correct values in my_Init

# Run:
* Optimization for N=1000 (T fixed), compare serial vs pint-oneshot time. 

## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
