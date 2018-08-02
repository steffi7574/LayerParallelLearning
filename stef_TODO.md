# TODO

* Opening layer + zero theta init seems to be a saddle point?
* Run optim with opening layer, instead of expand
    - check if n=32 is still good 
        - it is, if n=8 channels, use expanding opening layer
        - what if opening layer with sigma
    - check for n=512
* Use smoothReLu activation function:
  -> max(x,0) but smooth it out around zero with quadratic approximation.

* Alternative initialization of deeper networks: Linear interpolation of the solution of a trained smaller network. 

# Finite Differences testing
- expand opening layer by zeros, use n=4 width, init theta_open = random, theta=0.0, classW/Mu = random
    -> Gradient wrt theta_open, theta, classW, classMu is correct. 
- opening layer with sigma(KY+bias), test also for iter > 0 !!
    -> Gradient wrt theta_open and classW/Mu fits. 
       BUT: Gradient is zero wrt thetas!!! WHY???

## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search

# Read:
- Intel MKA -> Convolution, library
