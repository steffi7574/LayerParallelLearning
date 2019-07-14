#include <stdio.h>
#include "linalg.hpp"

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
  virtual void computeAscentDir(int k, double *gradient, double *ascentdir) = 0;

  /**
   * Update the BFGS memory (like s, y, rho, H0...)
   */
  virtual void updateMemory(int k, double *design, double *gradient) = 0;
};

class L_BFGS : public HessianApprox {
 protected:
  int M; /* Length of the l-bfgs memory (stages) */

  /* L-BFGS memory */
  double **s;           /* storing M (x_{k+1} - x_k) vectors */
  double **y;           /* storing M (\nabla f_{k+1} - \nabla f_k) vectors */
  double *rho;          /* storing M 1/y^Ts values */
  double H0;            /* Initial Hessian scaling factor */
  double *design_old;   /* Design at previous iteration */
  double *gradient_old; /* Gradient at previous iteration */

 public:
  L_BFGS(MPI_Comm comm, int dimN, /* Local design dimension */
         int stage);
  ~L_BFGS();

  void computeAscentDir(int k, double *gradient, double *ascentdir);

  void updateMemory(int k, double *design, double *gradient);
};

class BFGS : public HessianApprox {
 private:
  double *A;
  double *B;
  double *Hy;

 protected:
  double *s;
  double *y;
  double
      *Hessian; /* Storing the Hessian approximation (flattened: dimN*dimN) */
  double *design_old;   /* Design at previous iteration */
  double *gradient_old; /* Gradient at previous iteration */

 public:
  BFGS(MPI_Comm comm, int N);
  ~BFGS();

  void setIdentity();

  void computeAscentDir(int k, double *gradient, double *ascentdir);

  void updateMemory(int k, double *design, double *gradient);
};

/**
 * No second order: Use Identity for Hessian Approximation
 */
class Identity : public HessianApprox {
 public:
  Identity(MPI_Comm comm, int N);
  ~Identity();

  void computeAscentDir(int k, double *currgrad, double *ascentdir);

  void updateMemory(int k, double *design, double *gradient);
};
