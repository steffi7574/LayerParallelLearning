#include <stdio.h>
#include "linalg.hpp"

#pragma once

class Layer 
{
   protected:
      int     nweights;                    /* Number of weight parameters */
      double  dt;                          /* Step size for Layer update */
      double* weights;                     /* Weight matrix, faltened as a vector */
      double  bias;                        /* Bias */
      double  (*activation)(double x);     /* Pointer to 1D activation function */

   private:
      double* update;                      /* Temporary vector for computing linear transformation */

   public:

      int     nchannels;                   /* Width of the Layer (number of channels) */ 

      Layer();
      Layer(int    nChannels, 
            double (*Activ)(double x));
      ~Layer();

      /* Set the bias */
      void setBias(double Bias);

      /* Set pointer to the weight vector (matrix) */
      void setWeights(double* Weights);

      /* Set time step size */
      void setDt(double DT);

      /**
       * Forward propagation of an example through the layer 
       * In/Out: vector holding the current example data
       */
      void applyFWD(double* data);

};