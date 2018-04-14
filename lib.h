#ifndef LIB_H_INCLUDED
#define LIB_H_INCLUDED


/** 
 * Forward propagation 
 */
int
take_step(double* Y,
          double* K,
          int     ts,
          double  dt,
          int    *batch,
          int     nbatch,
          int     nchannels,
          int     parabolic);


/**
 * Activation function 
 */
double 
sigma(double x);        


/**
 * Return maximum of two doubles */
double 
max(double a,
    double b);

/**
 * Evaluate the loss functin
 */
double  
loss(double*  Y,
     double*  Ytarget,
     int     *batch,
     int      nbatch,
     int      nchannels);


/**
 * Relaxation term 
 */
double
regularization(double* theta,
                  int     ts,
                  double  dt,
                  int     ntime,
                  int     nchannels);

/**
 * Read data from file 
 */
int
read_data(char *filename, 
          double *var, 
          int size);

/**
 * Write data to file
 */
int
write_data(char *filename,
           double *var, 
           int size);

#endif // LIB_H_INCLUDED
