 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef LBFGS_H
#define LBFGS_H


class L_BFGS {

   int dimN;             /* Dimension of the gradient vector */
   int M;                /* Length of the l-bfgs memory */

   /* L-BFGS memory */
   double** s;     /* storing M (x_{k+1} - x_k) vectors */
   double** y;     /* storing M (\nabla f_{k+1} - \nabla f_k) vectors */
   double*  rho;    /* storing M 1/y^Ts values */
   double   H0;      /* Initial Hessian scaling factor */

   public:
      L_BFGS(int N, 
             int stage);        /* Constructor */
      ~L_BFGS();                       /* Destructor */

      /* Compute the BFGS descent direction */
      int compute_step(int     k, 
                       double* currgrad, 
                       double* step);

      /* Update the L-BFGS memory (s, y rho and H0) */
      int update_memory(int     k,
                        double* xnew,
                        double* xold,
                        double* gradnew,
                        double* gradold);

      /* Compute the dot-product of two vectors */
      double vecdot(double* x, 
                    double* y);                  
      
};

#endif