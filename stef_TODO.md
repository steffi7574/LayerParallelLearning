# TODO

* Use smoothReLu activation function:
  -> max(x,0) but smooth it out around zero with quadratic approximation.

* Alternative initialization of deeper networks: Linear interpolation of the solution of a trained smaller network. 


## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search

# Read:
- Intel MKA -> Convolution, library
