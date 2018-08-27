# TODO

## Program:
* Use Eigen for data management and linear algebra stuff. 
* Put activation and dactivation into a class / struct 
* Convolutional network

## Run:
* Optimization with nlayer=1024
* One-shot paramstudy

# Parameter study: Lessons Learned

* if **tanh** activation:
    - *Expand* input data to first layer using zeros
    - Initialization: 
            - theta:        random, factor 1e-3 or bigger
            - classifier:   zero
    - Regularization: 
            - tikhonov-term:    small, e.g. 1e-7
            - ddt-term:         small, e.g. 1e-7
            - class-term:       small, e.g. 1e-5, 1e-7

* if **ReLu** activation:
    - Use *opening layer* to map input data to first layer
    - Initialization:
            - theta:                zero
            - classifier:           random, factor 1e-3
            - theta opening layer:  random, factor 1e-3
    - Regularization:
            - tikhonov-term:    small, e.g. 1e-5 or 1e-7
            - ddt-term:         small, e.g. 1e-5 or 1e-7
            - class-term:       1e-3

* Rule of thumb for initialization: 
    loss(first iteration) = - log(P(GuessTheRightClass))


# Read:
* Intel MKA -> Convolution, library

## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
