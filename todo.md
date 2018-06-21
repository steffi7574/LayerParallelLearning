# TODO:

## Programming
* Implement the regularization term for W and mu
* If batch != examples: Initialize u->Ytrain with correct values in my_Init
* Read the classifier weights and bias from file ?

# Run:
* Optimization for N=1000 (T fixed), compare serial vs pint-oneshot time. 
* Log into LLNL cluster

## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
