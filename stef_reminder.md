# TODO

* Optimizer class implementation 

* Weights parametrization using splines




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


# 01|2019

* XBraid has been added as a git submodule. Run "git submodule update" after "git pull" to ensure that the correct xbraid commit is in place. 

* Batch optimization implementation. New config options:
  - **nbatch** decides on the number of elements in the batch (*nbatch* <= *ntraining*). *ntraining* now only determines the number of elements that are read from the training data file. In contrast, nbatch decides how many of those are propagated through the network. For validation, the number of propagated elements is still determined by *nvalidation*. 
  - **batch\_type**, switches between *deterministic* and *stochastic* batch selection. *deterministic* chooses the first *nbatch* elements from the data set and keeps this batch fixed during optimization. *stochastic* switches the batch elements in each optimization iteration by choosing a random subset (of size *nbatch*).
  - **stepsize\_type** decides on the stepsize selection method within design updates: *fixed* uses the initial stepsize for all design updates. *backtrackingLS* applies a backtracking line search. *oneoverk* sets the stepsize to 1/k where k is the current optimization iteration index (see e.g. Lars' lecture notes). 
 
   For vanilla SGD, choose 
     - *nbatch = 1*, and *ntraining* as big as possible
     - *batch_type = stochastic*
     - *stepsize\_type = oneoverk*
     - *hessian\_approx = Identity*

* Parallel implementation of LBFGS hessian approximation. 

* Added a basic python script for testing new implementations. It runs 6 test cases and compares the output with the reference. Currently only peaks case, using braid\_maxlevels being 1 or 10 and number of processors being 1, 2 or 5. In folder testing/ run *python testing.py* for testing all 6 cases or *python testing.py --help* for other configuration. 

