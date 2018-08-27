#include <stdlib.h>
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
      int dim_In;                          /* Dimension of incoming data */
      int dim_Out;                         /* Dimension of outgoing data */
      int dim_Bias;                        /* Dimension of the bias vector */

      int     index;                       /* Number of the layer */
      double  dt;                          /* Step size for Layer update */
      double* weights;                     /* Weight matrix, flattened as a vector */
      double* weights_bar;                 /* Derivative of the Weight matrix*/
      double* bias;                        /* Bias */
      double* bias_bar;                    /* Derivative of bias */
      double  (*activation)(double x);     /* the activation function */
      double  (*dactivation)(double x);    /* derivative of activation function */

   public:
      Layer();
      Layer(int     idx,
            int     dimI,
            int     dimO,
            int     dimB,
            double  deltaT,
            double (*Activ)(double x),
            double (*dActiv)(double x));     
      ~Layer();

      /* Set time step size */
      void setDt(double DT);

      /**
       * Initialize the layer, e.g. set initial weights and bias and gradients
       * Init random, scaled by a factor (gradient zero)
       */
      void initialize(double factor);

      /**
       * Forward propagation of an example 
       * In/Out: vector holding the current example data
       */
      virtual void applyFWD(double* data_In, 
                            double* data_Out) = 0;


      /**
       * Backward propagation of an example 
       * In:     data     - current example data
       * In/Out: data_bar - adjoint example data that is to be propagated backwards 
       */
      virtual void applyBWD(double* data_In,
                            double* data_Out,
                            double* data_In_bar,
                            double* data_Out_bar)=0;
};

/**
 * Layer consisting of dense weight matrix K \in R^{nxn}
 * Linear transformation is a matrix multiplication plus 1D bias: Ky + bias
 */
class DenseLayer : public Layer {

  public:
      DenseLayer(int     idx,
                 int     dimI,
                 int     dimO,
                 double  deltaT,
                 double (*Activ)(double x),
                 double (*dActiv)(double x));     
      ~DenseLayer();

      void applyFWD(double* data_In, 
                    double* data_Out);

      void applyBWD(double* data_In,
                    double* data_Out,
                    double* data_In_bar,
                    double* data_Out_bar);
};



/*
 * Opening layer that expands the data by zeros
 */
class OpenExpandZero : public Layer 
{
      public:
            OpenExpandZero(int     dimI,
                           int     dimO,
                           double  deltaT);
            ~OpenExpandZero();

            void applyFWD(double* data_In, 
                          double* data_Out);
      
            void applyBWD(double* data_In,
                          double* data_Out,
                          double* data_In_bar,
                          double* data_Out_bar);
};


/**
 * Classification layer
 */
class ClassificationLayer : public Layer
{
      public:
            ClassificationLayer(int idx,
                                int dimI,
                                int dimO,
                                double  deltaT);
            ~ClassificationLayer();

            void applyFWD(double* data_In, 
                          double* data_Out);
      
            void applyBWD(double* data_In,
                          double* data_Out,
                          double* data_In_bar,
                          double* data_Out_bar);
};