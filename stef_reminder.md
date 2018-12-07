# TODO

* Memory reduction:
    - single precision
    - sending / receiving a layer allocates buffer of size O(nchannels^2) even if that's way too big for the convolution layers.

* Weights parametrization using splines
* Remove opening layer from braid treatment, use it only with in my\_Init(). 


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
