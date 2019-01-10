#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "defs.hpp"

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
      virtual void computeAscentDir(int     k, 
                               MyReal* currgrad, 
                               MyReal* descdir) = 0;   

      /**
       * Update the BFGS memory (like s, y, rho, H0...)
       */
      virtual void updateMemory(int     k,
                                MyReal* xnew,
                                MyReal* xold,
                                MyReal* gradnew,
                                MyReal* gradold) = 0;

};


class L_BFGS : public HessianApprox {

   int M;                /* Length of the l-bfgs memory */

   /* L-BFGS memory */
   MyReal** s;     /* storing M (x_{k+1} - x_k) vectors */
   MyReal** y;     /* storing M (\nabla f_{k+1} - \nabla f_k) vectors */
   MyReal*  rho;    /* storing M 1/y^Ts values */
   MyReal   H0;      /* Initial Hessian scaling factor */

   public:
      L_BFGS(int N, 
             int stage);        /* Constructor */
      ~L_BFGS();                /* Destructor */

      void computeAscentDir(int     k, 
                             MyReal* currgrad, 
                             MyReal* descdir);

      void updateMemory(int     k,
                        MyReal* xnew,
                        MyReal* xold,
                        MyReal* gradnew,
                        MyReal* gradold);

      
};


class BFGS : public HessianApprox {

   private:
      MyReal* A;
      MyReal* B;
      MyReal* Hy;
   
   protected:
      MyReal* s;          
      MyReal* y; 
      MyReal* Hessian;   /* Storing the Hessian approximation (flattened: dimN*dimN) */

   public:
      BFGS(int N);
      ~BFGS();

      void setIdentity();                    

      void computeAscentDir(int     k, 
                             MyReal* currgrad, 
                             MyReal* descdir);

      void updateMemory(int     k,
                        MyReal* xnew,
                        MyReal* xold,
                        MyReal* gradnew,
                        MyReal* gradold);
};



/**
 * No second order: Use Identity for Hessian Approximation 
 */ 
class Identity : public HessianApprox{

   public: 
      Identity(int N);
      ~Identity();

      void computeAscentDir(int     k, 
                             MyReal* currgrad, 
                             MyReal* descdir);

      void updateMemory(int     k,
                        MyReal* xnew,
                        MyReal* xold,
                        MyReal* gradnew,
                        MyReal* gradold);


};
