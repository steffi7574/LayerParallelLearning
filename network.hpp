#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "util.hpp"
#pragma once


class Network
{
   protected:
      int     nlayers_global;       /* Total number of Layers of the network */
      int     nlayers_local;        /* Number of Layers in this network block */
      int     nchannels;            /* Width of the network */
      MyReal  dt;                   /* Time step size */
      MyReal  loss;                 /* Value of the loss function */
      MyReal  accuracy;             /* Accuracy of the network prediction (percentage of successfully predicted classes) */

      int     startlayerID;         /* ID of the first layer on that processor */
      int     endlayerID;           /* ID of the last layer on that processor */

      int     ndesign_loc;          /* Number of design vars of this network (local on this proc) */
      MyReal* design;               /* Local vector of design variables*/
      MyReal* gradient;             /* Local Gradient */

      Layer*  openlayer;            /* At first processor: openinglayer, else: NULL */
      Layer** layers;               /* Array of hidden layers and classification layer at last processor */
      Layer*  layer_left;           /* Copy of last layer of left-neighbouring processor */
      Layer*  layer_right;          /* Copy of first layer of right-neighbouring processor */
   
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
              MyReal deltaT,
              MyReal gamma_tik, 
              MyReal gamma_ddt, 
              MyReal gamma_class,
              MyReal Weight_open_init,
	        int    networkType,
	        int    type_openlayer);
     
      ~Network();

      /* Get number of channels */
      int getnChannels();

      /* Get global number of layers */
      int getnLayers();

      /* Get initial time step size */
      MyReal getDT();

      /* Get local storage index of the a layer */
      int getLocalID(int ilayer);

      /* Return value of the loss function */
      MyReal getLoss();

      /* Return accuracy value */
      MyReal getAccuracy();
 
      /* Return a pointer to the design vector */
      MyReal* getDesign();
       
      /* Return a pointer to the gradient vector */
      MyReal* getGradient();

      /* Get ID of first and last layer on this processor */
      int getStartLayerID();
      int getEndLayerID();

      /**
       *  Return number of design variables (local on this processor) */
      int getnDesignLocal();

      /** 
       * Return max. number of layer's ndesign on this processor 
       * excluding opening and classification layer 
       */
      int getnDesignLayermax();

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
      void initialize(MyReal Weight_open_init,
                      MyReal Weight_init,
                      MyReal Classification_init,
                      char   *datafolder,
                      char   *weightsopenfile,
                      char   *weightsclassificationfile);


      Layer* createLayer(int    index, 
                         int    nfeatures,
                         int    nclasses,
                         int    Activation,
                         MyReal gamma_tik,
                         MyReal gamma_ddt,
                         MyReal gamma_class,
                         MyReal weights_open_init,
			       int    NetworkType,
                         int    Type_OpenLayer);
     
        
      /* Replace the layer with one that is received from the left neighbouring processor */  
      void MPI_CommunicateNeighbours(MPI_Comm comm);

};

