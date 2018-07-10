# TODO

## Programming
* Check memory consumption for primal run and adjoint overhead. 
* Control output of optimization:
    - Loss function AND 
    - Objective function (= Loss + Regularization)

# Run:
* Optimization for N=1000 (T fixed), compare serial vs pint-oneshot time. 

## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
