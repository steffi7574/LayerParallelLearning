#ifndef LIB_H_INCLUDED
#define LIB_H_INCLUDED

#include "codi.hpp"
using namespace codi;


/** 
 * Forward propagation 
 */
template <typename myDouble>
int
take_step(myDouble *Y,
          myDouble *K,
          int          ts,
          double       dt,
          int         *batch,
          int          nbatch,
          int          nchannels,
          int          parabolic);


/**
 * Activation function 
 */
template <typename myDouble>
myDouble 
sigma(myDouble x);        


/**
 * Derivative of activation function 
 */
double
sigma_diff(double x);


/**
 * Return maximum of two doubles */
template <typename myDouble>
myDouble 
maximum(myDouble a,
        myDouble b);

/**
 * Evaluate the loss functin
 */
template <typename myDouble>
myDouble  
loss(myDouble    *Y,
     double      *Target,
     int         *batch,
     int          nbatch,
     myDouble    *class_W,
     myDouble    *class_mu,
     int          nclasses,
     int          nchannels);


/**
 * Relaxation term 
 */
template <typename myDouble>
myDouble
regularization_theta(myDouble* theta,
                     int          ts,
                     double       dt,
                     int          ntime,
                     int          nchannels);

/**
 * Read data from file 
 */
template <typename myDouble> 
int
read_data(char   *filename, 
          myDouble *var, 
          int     size);

/**
 * Write data to file
 */
template <typename myDouble>
int
write_data(char   *filename,
           myDouble *var, 
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
