# TODO

## Program:
* Use Eigen for data management and linear algebra stuff. 
* Branch localdesignstorage (distributedcontrol): 
    - Random initialization of the weights will be different across the proccesors. -> Difficult to compare runs with varying numbers of processors... -> Think!
    - Generalize MPI communication of layers for ANY Layer type! e.g. via #defines for each layer type or so. 
* Branch solveadjointwithxbraid:
    - Include DDT-regularization and its derivative. 


# Parameter study: Lessons Learned (n=32)

* if **tanh** activation:
    - *Expand* input data to first layer using zeros
    - Weights initialization: 
         * opening layer:    zero
         * general layer:   random, factor 1e-3 or bigger
         * classif layer:   zero
    - Regularization param: 
         * tikhonov-term:    small, e.g. 1e-7
         * ddt-term:         small, e.g. 1e-7
         * class-term:       small, e.g. 1e-5, 1e-7

* if **ReLu** activation:
    - Use *opening layer* to map input data to first layer
    - Initialization:
         * opening layer:   random, factor 1e-3
         * general layer:   zero
         * classif layer:   random, factor 1e-3
    - Regularization:
         * tikhonov-term:    small, e.g. 1e-5 or 1e-7
         * ddt-term:         small, e.g. 1e-5 or 1e-7
         * class-term:       1e-3

* Rule of thumb for initialization: 
    loss(first iteration) = - log(P(GuessTheRightClass))


# Read:
* **Stochastic approximation**: small batches, use SDG, no hessian approx
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
