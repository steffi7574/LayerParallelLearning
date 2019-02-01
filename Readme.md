#Latest Changes

## 01|2019

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



# Build
* Requires XBraid on branch 'solveadjointiwithxbraid', included as a git submodule:
    - git submodule init 
    - or git submodule update --init
* *make* should build both, xbraid library as well as dnn code. If not, type
    - cd xbraid
    - make braid
    - cd 
    - make

* git cloning with ssh:
    - The submodules don't really allow for ssh access
    - Remedy: modify .git/config file to point to an ssh version of the braid repo
