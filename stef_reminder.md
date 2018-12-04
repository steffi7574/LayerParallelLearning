# TODO

1. Remove opening layer from braid treatment, use it only with in my\_Init(). 
    TODO: 
        - Eval and check Regularization terms, in particular Tikhonov regul and Tikh_diff. 
        - Improve output of created network
        - Adapt pythonutil/config.py for new hiddenlayer option. 
        - Consistency of names for nlayers/nlayers_local/hiddenlayers 

2. Memory reduction:
    - sending / receiving a layer allocates buffer of size O(nchannels^2) even if that's way too big for the convolution layers.

3. Batch optimization: Allow for varying batches during optimization -> SGD and variants
    - Set new initial condition before braid\_Drive(). 

4. Weights parametrization using splines



# Parameter study: Lessons Learned (n=32, peaks example)

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

