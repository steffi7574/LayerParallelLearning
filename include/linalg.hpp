#include <math.h>
#include <mpi.h>
#include <stdio.h>
#pragma once

/**
 * Compute scalar product of two vectors xTy
 * In: dimension dimN
 *     vectors x and y of dimemsion dimN
 * Out: returns xTy
 */
double vecdot(int dimN, double *x, double *y);

/**
 * Parallel dot-product xTy, invokes an MPI_Allreduce call
 * In: dimension dimN
 *     vectors x and y of dimemsion dimN
 *     MPI communicator
 * Out: returns global xTy on all procs
 */
double vecdot_par(int dimN, double *x, double *y, MPI_Comm comm);

/**
 * Return the maximum value of a vector
 */
double vecmax(int dimN, double *x);

/**
 * Return the index of the maximum entry of the vector
 */
int argvecmax(int dimN, double *x);

/**
 * Computes square of the l2-norm of x
 */
double vecnormsq(int dimN, double *x);

/**
 * Parallel l2-norm computation, invokes an MPI_Allreduce x
 */
double vecnorm_par(int dimN, double *x, MPI_Comm comm);

/**
 * Copy a vector u into u_copy
 */
int vec_copy(int N, double *u, double *u_copy);

/**
 * Compute matrix x* y^T
 */
void vecvecT(int N, double *x, double *y, double *XYT);

/**
 * Compute Matrix-vector product Hx
 * In: dimension dimN
 *     square Matrix H (flattened into a vector)
 *     vector x
 * Out: H*x will be stored in Hx
 */
void matvec(int dimN, double *H, double *x, double *Hx);
