#ifndef LIB_H_INCLUDED
#define LIB_H_INCLUDED

#include "codi.hpp"
using namespace codi;


/** 
 * Forward propagation 
 */
int
take_step(RealReverse *Y,
          RealReverse *K,
          int          ts,
          double       dt,
          int         *batch,
          int          nbatch,
          int          nchannels,
          int          parabolic);


/**
 * Activation function 
 */
RealReverse 
sigma(RealReverse x);        


/**
 * Derivative of activation function 
 */
RealReverse
sigma_diff(RealReverse x);


/**
 * Return maximum of two doubles */
RealReverse 
max(RealReverse a,
    RealReverse b);

/**
 * Evaluate the loss functin
 */
RealReverse  
loss(RealReverse *Y,
     double      *Ytarget,
     int         *batch,
     int          nbatch,
     int          nchannels);


/**
 * Relaxation term 
 */
RealReverse
regularization(RealReverse* theta,
               int          ts,
               double       dt,
               int          ntime,
               int          nchannels);

/**
 * Read data from file 
 */
template <typename myType> 
int
read_data(char   *filename, 
          myType *var, 
          int     size);

/**
 * Write data to file
 */
template <typename myType>
int
write_data(char   *filename,
           myType *var, 
           int     size);

/**
 * CODI TYPE: Get the primal value 
 */
double 
getValue(RealReverse value);

/**
 * DOUBLE TYPE: Return value 
 */
double 
getValue(double value);


#endif // LIB_H_INCLUDED
