#include <stdio.h>
#include "linalg.hpp"

#pragma once

/**
 * Abstract base class for the network layers 
 * Subclasses implement
 *    - applyFWD: Forward propagation of data 
 *    - applyBWD: Backward propagation of data 
 */
class Layer 
{
   protected:
      double  dt;                          /* Step size for Layer update */
      double* weights;                     /* Pointer to Weight matrix, flattened as a vector */
      double* weights_bar;                 /* Pointer to Derivative of the Weight matrix*/
      double* bias;                        /* Pointer to Bias */
      double* bias_bar;                    /* Pointer to Derivative of bias */
      double  (*activation)(double x);     /* Pointer to the activation function */
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

      /* Set pointer to input data */
      virtual void setInputData(double* inputdata_ptr);

      /**
       * Forward propagation of an example 
       * In/Out: vector holding the current example data
       */
      virtual void applyFWD(double* data) = 0;


      /**
       * Backward propagation of an example 
       * In:     data     - current example data
       * In/Out: data_bar - adjoint example data that is to be propagated backwards 
       */
      virtual void applyBWD(double* data, 
                             double* data_bar) = 0;
};

/**
 * Layer consisting of dense weight matrix K \in R^{nxn}
 * Linear transformation is a matrix multiplication Ky
 */
class DenseLayer : public Layer {

  public:
      DenseLayer(int    nChannels, 
                 double (*Activ)(double x),
                 double (*dActiv)(double x));
      ~DenseLayer();

      void applyFWD(double* data);

      void applyBWD(double* data, 
                    double* data_bar);
};


/**
 * Opening layer: Maps the input data to the netword width (channels)
 */
class OpenLayer : public Layer{

   protected:
      int      nfeatures;      /* Number of features of the input data examples */
      double *inputData;      /* Pointer to the input data */

   public:
      OpenLayer(int nChannels,
                int nFeatures,
                double (*Activ)(double x),
                double (*dActiv)(double x));
      ~OpenLayer();

      void setInputData(double* inputdata_ptr);

      void applyFWD(double* data);

      void applyBWD(double* data, 
                    double* data_bar);
};