#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include "util.hpp"
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

      int     startlayerID;         /* ID of the first layer on that processor */
      int     endlayerID;           /* ID of the last layer on that processor */

      int     ndesign;              /* Number of design variables (local) */
      double* design;               /* Local vector of design variables*/
      double* gradient;             /* Local Gradient */

      Layer** layers;               /* Array of network layers */
      Layer*  layer_left;           /* Copy of last layer of right-neighbouring processor */
      Layer*  layer_right;          /* Copy of first layer of left-neighbouring processor */
   
   public: 
      enum networkType{DENSE, CONVOLUTIONAL}; /* Types of networks */

      Network();
      Network(int    nLayersGlobal,
              int    StartLayerID, 
              int    EndLayerID, 
              int    nFeatures,
              int    nClasses,
              int    nChannels, 
              int    Activation,
              double deltaT,
              double gamma_tik, 
              double gamma_ddt, 
              double gamma_class,
              double Weight_open_init,
	        int    networkType,
	        int    type_openlayer);
     
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

      /**
       * Get the layer at a certain layer index, i.e. a certain time step
       * Returns NULL, if this layer is not stored on this processor 
       */
      Layer* getLayer(int layerindex);


      /* 
       * Initialize the layer weights and biases:
       * Default: Scales random initialization from main with the given factors
       * If set, reads in opening and classification weights from file
       */
      void initialize(double Weight_open_init,
                      double Weight_init,
                      double Classification_init,
                      char   *datafolder,
                      char   *weightsopenfile,
                      char   *weightsclassificationfile);


      Layer* createLayer(int    index, 
                         int    nfeatures,
                         int    nclasses,
                         int    Activation,
                         double gamma_tik,
                         double gamma_ddt,
                         double gamma_class,
                         double weights_open_init,
			       int    NetworkType,
                         int    Type_OpenLayer);
     
        
      /* Replace the layer with one that is received from the left neighbouring processor */  
      void MPI_CommunicateNeighbours(MPI_Comm comm);

};

