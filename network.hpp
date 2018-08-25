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

      Layer*  openlayer;       /* First Layer of the network */
      Layer** layers;          /* Array of intermediat network layers */
      Layer*  endlayer;        /* Last layer of the network */

   public: 
      enum activation{ RELU, TANH};  /* Available activation functions */
      Network();
      Network(int    nLayers,
              int    nChannels, 
              int    nFeatures,
              int    nClasses,
              int    Activation,
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
                    double **labels,
                    double  deltat);

      /* ReLu Activation and derivative */
      static double ReLu_act(double x);
      static double dReLu_act(double x);

      /* tanh Activation and derivative */
      static double tanh_act(double x);
      static double dtanh_act(double x);

};

