# TODO

## Programming
* Make some nice ouput scripts!
* Control output of optimization:
    - Loss function AND 
    - Objective function (= Loss + Regularization)

* If batch != examples: 
    - Initialize `u->Ytrain` with correct values in `my_Initi`
    - Check `bach` and `batch_id` for validation core. 

# Run:
* Optimization for N=1000 (T fixed), compare serial vs pint-oneshot time. 

## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
