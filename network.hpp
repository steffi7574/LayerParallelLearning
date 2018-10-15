#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#pragma once


class Network
{
   protected:
      int     nlayers_global;       /* Total number of Layers of the network */
      int     nlayers_local;        /* Number of Layers in this network block */
      int     nchannels;            /* Width of the network */
      double  dt;                   /* Time step size */
      double  loss;                 /* Value of the loss function */
      double  accuracy;             /* Accuracy of the network prediction (percentage of successfully predicted classes) */
      double  gamma_tik;            /* Parameter for Tikhonov-regularization */
      double  gamma_ddt;            /* Parameter for ddt-regularization */
      double  gamma_class;          /* Parameter for Classification-regularization */

      int     startlayerID;         /* ID of the first layer on that processor */
      int     endlayerID;           /* ID of the last layer on that processor */

      int     ndesign;              /* Number of design variables (local) */
      double* design;               /* Local vector of design variables*/
      double* gradient;             /* Local Gradient */
      double (*activ_ptr)(double x);    /* Activation function pointer */
      double (*dactiv_ptr)(double x);   /* Derivative of activation function */

   public: 
      Layer** layers;               /* Array of network layers */
      enum activation{TANH, RELU, SMRELU}; /* Available activation functions */

      Network();
      Network(int    nLayers,
              int    nChannels, 
              int    Activation,
              double deltaT,
              double Gamma_tik, 
              double Gamma_ddt,
              double Gamma_class);
      ~Network();

      /* Get number of channels */
      int getnChannels();

      /* Get global number of layers */
      int getnLayers();

      /* Get initial time step size */
      double getDT();

      /* Get local storage index of the a layer */
      int getLocalID(int ilayer);

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


      void createLayers(int    StartLayerID, 
                        int    EndLayerID, 
                        int    nFeatures,
                        int    nClasses,
                        double Weight_init,
                        double Weight_open_init,
                        double Classification_init);

//       /**
//        * Forward propagation through the network. Evaluates loss and accuracy at last layer. 
//        * In: - number of examples
//        *     - Pointer to input data, is NULL for all but the first processor!
//        *     - Pointer to data labels, is NULL for all but the last processor!
//        */
//       void applyFWD(int     nexamples,
//                     double **examples,
//                     double **labels);


//       /**
//        * Returns the regularization term 
//        */
//       double evalRegularization();
      
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

