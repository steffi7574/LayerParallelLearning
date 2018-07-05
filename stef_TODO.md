# TODO

## Programming
* Load and prepare training data:
    - load Ytrain and class labels C from file
    - expand those to the width of the network. 
* Control output of optimization:
    - Loss function 
    - Objective function (= Loss + Regularization)
    - number of non-zeros(Cdata - Cpredicted) / 2, div by number of examples -> accuracy. 

* If batch != examples: Initialize u->Ytrain with correct values in my_Init

# Run:
* Optimization for N=1000 (T fixed), compare serial vs pint-oneshot time. 

## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
