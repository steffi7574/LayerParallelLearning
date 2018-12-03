#include <stdio.h>
#include <math.h>
#include "defs.hpp"
#pragma once

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
 * Return square of the l2-norm of the vector x
 */
MyReal vec_normsq(int    dimN,
                  MyReal *x);


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
