#include <stdio.h>
#include "linalg.hpp"

#pragma once

class Layer 
{
   protected:
      double  dt;                          /* Step size for Layer update */
      double* weights;                     /* Pointer to Weight matrix, flattened as a vector */
      double* weights_bar;                 /* Pointer to Derivative of the Weight matrix*/
      double* bias;                        /* Pointer to Bias */
      double* bias_bar;                    /* Pointer to Derivative of bias */
      double  (*activation)(double x);     /* Pointer to 1D activation function */
      double  (*dactivation)(double x);    /* Pointer to derivative of activation function */
      double* update;                      /* Temporary vector for computing linear transformation */
      double* update_bar;                  /* Temporary vector used in backpropagation */

   public:
      int     nchannels;                   /* Width of the Layer (number of channels) */ 
      Layer();
      Layer(int    nChannels, 
            double (*Activ)(double x),
            double  (*dActiv)(double x));     
      ~Layer();

      /* Set the bias */
      void setBias(double* bias_ptr);

      /* Set the derivative of bias */
      void setBias_bar(double* bias_bar_ptr);

      /* Set pointer to the weight vector (matrix) */
      void setWeights(double* weights_ptr);

      /* Set pointer to weight gradient */
      void setWeights_bar(double* weights_bar_ptr);

      /* Set time step size */
      void setDt(double DT);

      /**
       * Forward propagation of an example 
       * In/Out: vector holding the current example data
       */
      void applyFWD(double* data);


      /**
       * Backward propagation of an example 
       * In:     data     - current example data
       * In/Out: data_bar - adjoint example data that is to be propagated backwards 
       */
      void applyBWD(double* data, 
                    double* data_bar);


};