#include <stdio.h>
#pragma once

/**
 * Compute Matrix-vector product Hx
 * In: dimension dimN
 *     square Matrix H (flattened into a vector)
 *     vector x
 * Out: H*x will be stored in Hx
 */
void matvec(int     dimN,
            double* H, 
            double* x,
            double* Hx);



/**
 * Compute scalar product of two vectors xTy
 * In: dimension dimN
 *     vectors x and y of dimemsion dimN
 * Out: returns xTy
 */
double vecdot(int     dimN,
              double* x,
              double* y);

/**
 * Return the maximum value of a vector 
 */
double vecmax(int     dimN,
              double* x);


/**
 * Return the index of the maximum entry of the vector 
 */
int argvecmax(int     dimN,
              double* x);