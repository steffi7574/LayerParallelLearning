#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#pragma once

class HessianApprox {

   protected:
      int dimN;             /* Dimension of the gradient vector */

   public:
      HessianApprox();
      virtual ~HessianApprox();

      /**
       * Compute the BFGS descent direction 
       */
      virtual void computeDescentDir(int     k, 
                               double* currgrad, 
                               double* descdir) = 0;   

      /**
       * Update the BFGS memory (like s, y, rho, H0...)
       */
      virtual void updateMemory(int     k,
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

      void computeDescentDir(int     k, 
                             double* currgrad, 
                             double* descdir);

      void updateMemory(int     k,
                        double* xnew,
                        double* xold,
                        double* gradnew,
                        double* gradold);

      
};


class BFGS : public HessianApprox {

   private:
      double* A;
      double* B;
      double* Hy;
   
   protected:
      double* s;          
      double* y; 
      double* Hessian;   /* Storing the Hessian approximation (flattened: dimN*dimN) */

   public:
      BFGS(int N);
      ~BFGS();

      void setIdentity();                    

      void computeDescentDir(int     k, 
                             double* currgrad, 
                             double* descdir);

      void updateMemory(int     k,
                        double* xnew,
                        double* xold,
                        double* gradnew,
                        double* gradold);
};



/**
 * No second order: Use Identity for Hessian Approximation 
 */ 
class Identity : public HessianApprox{

   public: 
      Identity(int N);
      ~Identity();

      void computeDescentDir(int     k, 
                             double* currgrad, 
                             double* descdir);

      void updateMemory(int     k,
                        double* xnew,
                        double* xold,
                        double* gradnew,
                        double* gradold);


};
