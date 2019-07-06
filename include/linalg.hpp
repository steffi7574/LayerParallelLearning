#include <stdio.h>
#include <math.h>
#include "defs.hpp"
#include <mpi.h>
#pragma once


/**
 * Compute scalar product of two vectors xTy
 * In: dimension dimN
 *     vectors x and y of dimemsion dimN
 * Out: returns xTy
 */
MyReal vecdot(int     dimN,
              MyReal* x,
              MyReal* y);


/**
 * Parallel dot-product xTy, invokes an MPI_Allreduce call 
 * In: dimension dimN
 *     vectors x and y of dimemsion dimN
 *     MPI communicator 
 * Out: returns global xTy on all procs
 */
MyReal vecdot_par(int     dimN,
                  MyReal* x,
                  MyReal* y,
                  MPI_Comm comm);


/**
 * Return the maximum value of a vector 
 */
MyReal vecmax(int     dimN,
              MyReal* x);


/**
 * Return the index of the maximum entry of the vector 
 */
int argvecmax(int     dimN,
              MyReal* x);


/**
 * Computes square of the l2-norm of x
 */
MyReal vecnormsq(int      dimN,
                 MyReal   *x);

/**
 * Parallel l2-norm computation, invokes an MPI_Allreduce x
 */
MyReal vecnorm_par(int      dimN,
                   MyReal   *x,
                   MPI_Comm comm);


/**
 * Copy a vector u into u_copy 
 */
int vec_copy(int N, 
             MyReal* u, 
             MyReal* u_copy);


/**
 * Compute matrix x* y^T
 */
void vecvecT(int N,
             MyReal* x,
             MyReal* y,
             MyReal* XYT);

/**
 * Compute Matrix-vector product Hx
 * In: dimension dimN
 *     square Matrix H (flattened into a vector)
 *     vector x
 * Out: H*x will be stored in Hx
 */
void matvec(int     dimN,
            MyReal* H, 
            MyReal* x,
            MyReal* Hx);


