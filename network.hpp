#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#pragma once

class Network
{
   protected:
      int     nlayers;         /* Total number of Layers */
      int     nexamples;       /* Number of input data examples */

      Layer*  openlayer;       /* First Layer of the network */
      Layer** layers;          /* Array of intermediat network layers */
      Layer*  endlayer;        /* Last layer of the network */

      enum activation{ RELU, TANH};  /* Available activation functions */

   public: 
      Network();
      Network(int    nLayers,
              int    nExamples,
              int    nChannels, 
              int    nFeatures,
              int    nClasses,
              int    Activation,
              double Weight_init,
              double Weight_open_init,
              double Classification_init);
      ~Network();

      /* ReLu Activation and derivative */
      static double ReLu_act(double x);
      static double dReLu_act(double x);

      /* tanh Activation and derivative */
      static double tanh_act(double x);
      static double dtanh_act(double x);

};

