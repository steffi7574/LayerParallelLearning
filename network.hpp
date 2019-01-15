#include <stdio.h>
#include "layer.hpp"
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "config.hpp"
#include "util.hpp"
#include "dataset.hpp"
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

      int     ndesign_global;        /* Global number of design vars  */
      int     ndesign_local;         /* Number of design vars of this local network block  */
      int     ndesign_layermax;      /* Max. number of design variables of all hidden layers */

      MyReal* design;               /* Local vector of design variables*/
      MyReal* gradient;             /* Local Gradient */

      Layer*  openlayer;            /* At first processor: openinglayer, else: NULL */
      Layer** layers;               /* Array of hidden layers (includes classification layer at last processor */
      Layer*  layer_left;           /* Copy of last layer of left-neighbouring processor */
      Layer*  layer_right;          /* Copy of first layer of right-neighbouring processor */
   
   public: 

      Network();
      Network(int     StartLayerID, 
              int     EndLayerID,
              Config* config); 
    
      ~Network();

      /* Get number of channels */
      int getnChannels();

      /* Get global number of layers */
      int getnLayersGlobal();

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
       *  Return number of design variables (local on this processor or global) */
      int getnDesignLocal();
      int getnDesignGlobal();

      /** 
       * Compute max. number of layer's ndesign on this processor 
       * excluding opening and classification layer 
       */
      int computeLayermax();

      /* Return ndesign_layermax */
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
      void initialize(Config *config);


      Layer* createLayer(int     index, 
                         Config *config);
    
        
      /* Replace the layer with one that is received from the left neighbouring processor */  
      void MPI_CommunicateNeighbours(MPI_Comm comm);

     /**
       * Applies the classification and evaluates loss/accuracy 
       */
      void evalClassification(DataSet* data, 
                              MyReal** state,
                              MyReal*  loss_ptr, 
                              MyReal*  accuracy_ptr,
                              int      output);

      /**
       * On classification layer: derivative of evalClassification 
       */
      void evalClassification_diff(DataSet* data, 
                                   MyReal** primalstate,
                                   MyReal** adjointstate,
                                   int      compute_gradient);


      /**
       * Update the network design parameters: new_design = old_design + stepsize * direction
       */
      void updateDesign(MyReal   stepsize,
                        MyReal  *direction,
                        MPI_Comm comm);
};

