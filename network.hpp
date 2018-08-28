#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#pragma once


class Network
{
   protected:
      int     nlayers;         /* Total number of Layers */
      int     nchannels;       /* Width of the network */
      double  dt;              /* Time step size */
      double  loss;            /* Value of the loss function */
      double  accuracy;        /* Accuracy of the network prediction (percentage of successfully predicted classes) */

      double* state_old;              /* Auxiliary: holding old state at previous layer */
      double* state_curr;             /* Auxiliary: holding current state at a layer */

   public: 
      Layer** layers;                 /* Array of network layers */
      enum activation{ RELU, TANH};   /* Available activation functions */

      Network();
      Network(int    nLayers,
              int    nChannels, 
              int    nFeatures,
              int    nClasses,
              int    Activation,
              double deltaT,
              double Weight_init,
              double Weight_open_init,
              double Classification_init);
      ~Network();

      /* Get dimensions */
      int getnChannels();
      int getnLayers();

      /* Other get functions */
      double getLoss();
      double getAccuracy();

      /**
       * Sets the state of the network to the given data of dimensions dimN.
       * Requires dimN <= nchannels! Fills the rest with zeros.
       */
      void setState(int     dimN, 
                    double* data);

     /**
       * Sets the old state of the network to the given data of dimensions dimN.
       * Requires dimN <= nchannels! Fills the rest with zeros.
       */
      void setState_Old(int     dimN,
                        double* data);


      /**
       * Forward propagation through the network. Evaluates loss and accuracy at last layer. 
       * In: - number of examples
       *     - Pointer to input data, is NULL for all but the first processor!
       *     - Pointer to data labels, is NULL for all but the last processor!
       */
      void applyFWD(int     nexamples,
                    double **examples,
                    double **labels);


      /**
       * Returns the regularization term 
       */
      double evalRegularization(double gamma_tik,
                                         double gamma_ddt);
      
      /**
       * Regularization for the time-derivative of the layer weights
       */
      double evalRegulDDT(Layer* layer_old, 
                          Layer* layer_curr);            

      /* ReLu Activation and derivative */
      static double ReLu_act(double x);
      static double dReLu_act(double x);

      /* tanh Activation and derivative */
      static double tanh_act(double x);
      static double dtanh_act(double x);

};

