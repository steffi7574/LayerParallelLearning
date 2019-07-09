#include "defs.hpp"
#include "linalg.hpp"
#include <stdio.h>

#pragma once

class HessianApprox {

protected:
  int dimN;         /* Dimension of the gradient vector */
  MPI_Comm MPIcomm; /* MPI communicator for parallel L-BFGS updates */

public:
  HessianApprox(MPI_Comm comm);
  virtual ~HessianApprox();

  /**
   * Compute the BFGS descent direction
   */
  virtual void computeAscentDir(int k, MyReal *gradient, MyReal *ascentdir) = 0;

  /**
   * Update the BFGS memory (like s, y, rho, H0...)
   */
  virtual void updateMemory(int k, MyReal *design, MyReal *gradient) = 0;
};

class L_BFGS : public HessianApprox {

protected:
  int M; /* Length of the l-bfgs memory (stages) */

  /* L-BFGS memory */
  MyReal **s;           /* storing M (x_{k+1} - x_k) vectors */
  MyReal **y;           /* storing M (\nabla f_{k+1} - \nabla f_k) vectors */
  MyReal *rho;          /* storing M 1/y^Ts values */
  MyReal H0;            /* Initial Hessian scaling factor */
  MyReal *design_old;   /* Design at previous iteration */
  MyReal *gradient_old; /* Gradient at previous iteration */

public:
  L_BFGS(MPI_Comm comm, int dimN, /* Local design dimension */
         int stage);
  ~L_BFGS();

  void computeAscentDir(int k, MyReal *gradient, MyReal *ascentdir);

  void updateMemory(int k, MyReal *design, MyReal *gradient);
};

class BFGS : public HessianApprox {

private:
  MyReal *A;
  MyReal *B;
  MyReal *Hy;

protected:
  MyReal *s;
  MyReal *y;
  MyReal
      *Hessian; /* Storing the Hessian approximation (flattened: dimN*dimN) */
  MyReal *design_old;   /* Design at previous iteration */
  MyReal *gradient_old; /* Gradient at previous iteration */

public:
  BFGS(MPI_Comm comm, int N);
  ~BFGS();

  void setIdentity();

  void computeAscentDir(int k, MyReal *gradient, MyReal *ascentdir);

  void updateMemory(int k, MyReal *design, MyReal *gradient);
};

/**
 * No second order: Use Identity for Hessian Approximation
 */
class Identity : public HessianApprox {

public:
  Identity(MPI_Comm comm, int N);
  ~Identity();

  void computeAscentDir(int k, MyReal *currgrad, MyReal *ascentdir);

  void updateMemory(int k, MyReal *design, MyReal *gradient);
};
