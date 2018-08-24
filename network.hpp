#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#pragma once

class Network
{
   protected:
      int     nlayers;         /* Total number of Layers */
      int     nexamples;       /* Number of data elements */

      Layer*  openlayer;       /* First Layer of the network */
      Layer** layers;          /* Array of intermediat network layers */
      Layer*  endlayer;        /* Last layer of the network */

      enum activation{ RELU, TANH};  /* Available activation functions */

   public: 
      Network();
      Network(int    nLayers,
              int    nexamples,
              int    nchannels, 
              int    nfeatures,
              int    nclasses,
              int    activation,
              double weight_init,
              double weight_open_init,
              double classification_init);
      ~Network();

      /* ReLu Activation and derivative */
      static double ReLu_act(double x);
      static double dReLu_act(double x);

      /* tanh Activation and derivative */
      static double tanh_act(double x);
      static double dtanh_act(double x);

};

