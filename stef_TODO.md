# TODO

* use 8 channels instead of 4
* Opening layer: sigma(K * data + bias), where K is a 8x2 matrix 
* Initialize K and bias with random matrix (gaußian, scale by say 1e-2)
* Initialize classifier W and mu with random matrix!
* Check: Loss of initial guess should be at the order of -log(P(RichtigeKlasseWirdGeraten)), i.e. -log(20%)
* Use smoothReLu activation function:
  -> max(x,0) but smooth it out around zero with quadratic approximation.

* Alternative initialization of deeper networks: Linear interpolation of the solution of a trained smaller network. 


## Algorithms
* **Stochastic approximation**: small batches, use SDG, no hessian approx, minmize the expectation 
* **Stochastic Average Approximation (SAA)**: big batches, BFGS with line-search
