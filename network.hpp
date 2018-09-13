#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#pragma once


class Network
{
   protected:
      int     nlayers;              /* Total number of Layers */
      int     nchannels;            /* Width of the network */
      double  dt;                   /* Time step size */
      double  loss;                 /* Value of the loss function */
      double  accuracy;             /* Accuracy of the network prediction (percentage of successfully predicted classes) */
      double  gamma_ddt;            /* Parameter for ddt-regularization */

      int     ndesign;              /* Number of design variables */
      double* design;               /* Vector of all design variables (weights & biases at all layers) */
      double* gradient;             /* Gradient */

   public: 
      Layer** layers;               /* Array of network layers */
      enum activation{TANH, RELU, SMRELU}; /* Available activation functions */

      Network();
      Network(int    nLayers,
              int    nChannels, 
              int    nFeatures,
              int    nClasses,
              int    Activation,
              double deltaT,
              double Weight_init,
              double Weight_open_init,
              double Classification_init,
              double Gamma_tik, 
              double Gamma_ddt,
              double Gamma_class);
      ~Network();

      /* Get dimensions */
      int getnChannels();
      int getnLayers();

      /* Return value of the loss function */
      double getLoss();

      /* Return accuracy value */
      double getAccuracy();
 
      /* Return a pointer to the design vector */
      double* getDesign();
       
      /* Return a pointer to the gradient vector */
      double* getGradient();

      /**
       *  Returns the total number of design variables (weights and biases at all layers) */
      int getnDesign();


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
      double evalRegularization();
      
      /**
       * Regularization for the time-derivative of the layer weights
       */
      double evalRegulDDT(Layer* layer_old, 
                          Layer* layer_curr);            

      /**
       * Derivative of ddt-regularization term 
       */
      void evalRegulDDT_diff(Layer* layer_old, 
                             Layer* layer_curr,
                             double regul_bar);


      /* ReLu Activation and derivative */
      static double ReLu_act(double x);
      static double dReLu_act(double x);
        
      /* Smooth ReLu activation: Uses a quadratic approximation around zero (range: default 0.1) */
      static double SmoothReLu_act(double x);
      static double dSmoothReLu_act(double x);

      /* tanh Activation and derivative */
      static double tanh_act(double x);
      static double dtanh_act(double x);

};

