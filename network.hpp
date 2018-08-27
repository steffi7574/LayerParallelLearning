#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#pragma once


class Network
{
   private:
      double* state_curr;      /* Auxiliary: holding current state at a layer */
      double* state_old;       /* Auxiliary: holding old state at previous layer */
      double* state_final;     /* Auxiliary: State after last layer (after classiication) */

   protected:
      int     nlayers;         /* Total number of Layers */
      int     nchannels;       /* Width of the network */
      int     nclasses;        /* Number of classes */
      double  dt;              /* Time step size */
      double  gamma_tik;       /* Parameter for tikhonov regularization */
      double  gamma_ddt;       /* Parameter for ddt-regularization */

      Layer*  openlayer;       /* First Layer of the network */
      Layer** layers;          /* Array of intermediat network layers */
      Layer*  endlayer;        /* Last layer of the network */
      double  loss;            /* Value of the loss function */
      double  accuracy;        /* Accuracy of the network prediction (percentage of successfully predicted classes) */

   public: 
      enum activation{ RELU, TANH};  /* Available activation functions */
      Network();
      Network(int    nLayers,
              int    nChannels, 
              int    nFeatures,
              int    nClasses,
              int    Activation,
              double deltaT,
              double gammaTIK,
              double gammaDDT,
              double Weight_init,
              double Weight_open_init,
              double Classification_init);
      ~Network();

      /**
       * Forward propagation through the network
       * In: - number of examples
       *     - Pointer to input data, is NULL for all but the first processor!
       *     - Pointer to data labels, is NULL for all but the last processor!
       *     - time step size 
       */
      void applyFWD(int     nexamples,
                    double **examples,
                    double **labels);

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

