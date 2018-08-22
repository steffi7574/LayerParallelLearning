 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#pragma once

class HessianApprox {
/* HessianApprox is an abstract class (due to the =0 behind at least on of the functions). This means, that no instance of this class is created. Because some of the functions (the once with =0) are not implemented for this base class, but only for the derived classes. */

   protected:
      int dimN;             /* Dimension of the gradient vector */

   public:
      HessianApprox();
      HessianApprox(int N);
      ~HessianApprox();

      /* Compute the dot-product of two vectors */
      double vecdot(double* x, 
                    double* y);                  

      /* Compute matrix vector product */
      int matvec(double* H, 
                 double* x,
                 double* Hx);                  


      virtual int compute_step(int     k, 
                               double* currgrad, 
                               double* step) = 0;   

      virtual int update_memory(int     k,
                                double* xnew,
                                double* xold,
                                double* gradnew,
                                double* gradold) = 0;

};


class L_BFGS : public HessianApprox {

   int M;                /* Length of the l-bfgs memory */

   /* L-BFGS memory */
   double** s;     /* storing M (x_{k+1} - x_k) vectors */
   double** y;     /* storing M (\nabla f_{k+1} - \nabla f_k) vectors */
   double*  rho;    /* storing M 1/y^Ts values */
   double   H0;      /* Initial Hessian scaling factor */

   public:
      L_BFGS(int N, 
             int stage);        /* Constructor */
      ~L_BFGS();                /* Destructor */

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

      
};


class BFGS : public HessianApprox {
   
   double* s;          
   double* y; 
   double* Hessian;   /* Storing the Hessian approximation (flattened: dimN*dimN) */

   public:
      BFGS(int N);
      ~BFGS();

      int setIdentity();                    

      int compute_step(int     k, 
                       double* currgrad, 
                       double* step);

      int update_memory(int     k,
                        double* xnew,
                        double* xold,
                        double* gradnew,
                        double* gradold);
};
