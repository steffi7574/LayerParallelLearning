// Copyright
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Underlying paper:
//
// Layer-Parallel Training of Deep Residual Neural Networks
// S. Guenther, L. Ruthotto, J.B. Schroder, E.C. Czr, and N.R. Gauger
//
// Download: https://arxiv.org/pdf/1812.04352.pdf
//
#include "network.hpp"
#include <assert.h>

Network::Network(MPI_Comm Comm) {
  nlayers_global = 0;
  nlayers_local = 0;
  nchannels = 0;
  dt = 0.0;
  loss = 0.0;
  accuracy = 0.0;

  startlayerID = 0;
  endlayerID = 0;

  ndesign_local = 0;
  ndesign_global = 0;
  ndesign_layermax = 0;

  design = NULL;
  gradient = NULL;

  layers = NULL;
  layer_left = NULL;
  layer_right = NULL;

  comm = Comm;
  MPI_Comm_rank(comm, &mpirank);
}

void Network::createLayerBlock(int StartLayerID, int EndLayerID, Config *config, 
                               int current_nhiddenlayers) {
  /* Initilizize */
  startlayerID = StartLayerID;
  endlayerID = EndLayerID;
  nlayers_local = endlayerID - startlayerID + 1;
  nlayers_global = current_nhiddenlayers+2; // hidden + opening + classification
  nchannels = config->nchannels;
  dt = (config->T) / (MyReal)(current_nhiddenlayers);

  ndesign_local = 0;
  int mylayermax = 0;


  /* Create vector of layers on this processor */
  layers = new Layer *[nlayers_local];  
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) {

    /* Create a layer */
    Layer* newlayer = createLayer(ilayer, config);
    ndesign_local += newlayer->getnDesign();
    // printf("creating hidden/class layer %d/%d, ndesign_local%d\n", ilayer,
    // nlayers_local, layers[storeID]->getnDesign());

    /* Update layermax */
    if (newlayer->getnDesign() > mylayermax && ilayer < nlayers_global - 2) 
      mylayermax = newlayer->getnDesign();

    /* Store the new layer in the layer vector */
    layers[getLocalID(ilayer)] = newlayer;
  }

  /* Allocate memory for network design and gradient variables */
  design = new MyReal[ndesign_local];
  gradient = new MyReal[ndesign_local];

  /* Set the memory locations for all layers */
  int istart = 0;
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) 
  {
    getLayer(ilayer)->setMemory(&(design[istart]), &(gradient[istart]));
    istart += getLayer(ilayer)->getnDesign();
  }

  /* Communicate global ndesign and layermax */
  MPI_Allreduce(&ndesign_local, &ndesign_global, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&mylayermax, &ndesign_layermax, 1, MPI_INT, MPI_MAX, comm);

  /* Create left and right neighbouring layer */
  int leftID = startlayerID - 1;
  int rightID = endlayerID + 1;
  layer_left = createLayer(leftID, config);
  layer_right = createLayer(rightID, config);

  /* Allocate neighbouring layer's design, if exist on this proc */
  if (layer_left != NULL) {
    MyReal *left_design = new MyReal[layer_left->getnDesign()];
    MyReal *left_gradient = new MyReal[layer_left->getnDesign()];
    layer_left->setMemory(left_design, left_gradient);
  }
  if (layer_right != NULL) {
    MyReal *right_design = new MyReal[layer_right->getnDesign()];
    MyReal *right_gradient = new MyReal[layer_right->getnDesign()];
    layer_right->setMemory(right_design, right_gradient);
  }
}

Network::~Network() {

  /* Delete the layers */
  for (int ilayer = 0; ilayer < nlayers_local; ilayer++) {
    delete layers[ilayer];
  }
  delete[] layers;

  /* Delete design and gradient */
  delete[] design;
  delete[] gradient;

  /* Delete neighbouring layer information */
  if (layer_left != NULL) {
    delete[] layer_left->getWeights();
    delete[] layer_left->getWeightsBar();
    delete layer_left;
  }

  if (layer_right != NULL) {
    delete[] layer_right->getWeights();
    delete[] layer_right->getWeightsBar();
    delete layer_right;
  }
}

int Network::getnChannels() { return nchannels; }

int Network::getnLayersGlobal() { return nlayers_global; }

MyReal Network::getDT() { return dt; }

int Network::getLocalID(int ilayer) {
  int idx = ilayer - startlayerID;
  return idx;
}

MyReal Network::getLoss() { return loss; }

MyReal Network::getAccuracy() { return accuracy; }

int Network::getnDesignLocal() { return ndesign_local; }

int Network::getnDesignGlobal() { return ndesign_global; }

MyReal *Network::getDesign() { return design; }

MyReal *Network::getGradient() { return gradient; }

int Network::getStartLayerID() { return startlayerID; }
int Network::getEndLayerID() { return endlayerID; }

MPI_Comm Network::getComm() { return comm; }

Layer *Network::createLayer(int index, Config *config) {
  Layer *layer = 0;
  if (index == -1)  // Opening layer
  {
    switch (config->network_type) {
      case DENSE:
        if (config->weights_open_init == 0.0) {
          layer = new OpenExpandZero(config->nfeatures, nchannels);
        } else {
          layer = new OpenDenseLayer(config->nfeatures, nchannels,
                                     config->activation, config->gamma_tik);
        }
        break;
      case CONVOLUTIONAL:
        /**< (Weight_open_init == 0.0) not needed for convolutional layers*/
        if (config->openlayer_type == 0) {
          layer = new OpenConvLayer(config->nfeatures, nchannels);
        } else if (config->openlayer_type == 1) {
          layer = new OpenConvLayerMNIST(config->nfeatures, nchannels);
        }
        break;
    }
  } else if (0 <= index && index < nlayers_global - 2)  // Intermediate layer
  {
    switch (config->network_type) {
      case DENSE:
        layer =
            new DenseLayer(index, nchannels, nchannels, dt, config->activation,
                           config->gamma_tik, config->gamma_ddt);
        break;
      case CONVOLUTIONAL:
        // TODO: Fix
        int convolution_size = 3;
        layer =
            new ConvLayer(index, nchannels, nchannels, convolution_size,
                          nchannels / config->nfeatures, dt, config->activation,
                          config->gamma_tik, config->gamma_ddt);
        break;
    }
  } else if (index == nlayers_global - 2)  // Classification layer
  {
    layer = new ClassificationLayer(index, nchannels, config->nclasses,
                                    config->gamma_class);
  } else {
    layer = NULL;
  }

  return layer;
}

Layer *Network::getLayer(int layerindex) {
  Layer *layer = NULL;

  /* Get the layer, if stored */
  if (layerindex == startlayerID - 1) {
    layer = layer_left;
  } else if (startlayerID <= layerindex && layerindex <= endlayerID) {
    layer = layers[getLocalID(layerindex)];
  } else if (layerindex == endlayerID + 1) {
    layer = layer_right;
  }

  return layer;
}

int Network::getnDesignLayermax() { return ndesign_layermax; }


void Network::setDesignRandom(MyReal factor_open, MyReal factor_hidden, MyReal factor_classification) {
  MyReal factor;
  MyReal *design_init=NULL;

  /* Create a random vector (do it on one processor for scaling test) */
  if (mpirank == 0) {
    srand(1.0);
    design_init = new MyReal[ndesign_global];
    for (int i = 0; i < ndesign_global; i++) {
      design_init[i] = (MyReal)rand() / ((MyReal)RAND_MAX);
    }
  }
  /* Scatter random vector to local design for all procs */
  MPI_ScatterVector(design_init, design, ndesign_local, 0, comm);

  /* Scale the weights and reset the gradien */
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) {
    if (ilayer == -1){  // opening layer
      factor = factor_open;
    } else if (ilayer == nlayers_global - 2) { // classification layer
      factor = factor_classification;
    } else { // hidden layer
      factor = factor_hidden;
    }
    vec_scale(getLayer(ilayer)->getnDesign(), factor, getLayer(ilayer)->getWeights());
    vec_setZero(getLayer(ilayer)->getnDesign(), getLayer(ilayer)->getWeightsBar());
  }

  /* Communicate the neighbours across processors */
  MPI_CommunicateNeighbours();

  if (mpirank == 0) delete[] design_init;

}


void Network::setDesignFromFile(const char* datafolder, const char* openingfilename, const char* hiddenfilename, const char* classificationfilename) {

  char filename[255];

  /* if set, overwrite opening design from file */
  if (strcmp(openingfilename, "NONE") != 0) {
    sprintf(filename, "%s/%s", datafolder, openingfilename);
    read_vector(filename, getLayer(-1)->getWeights(), getLayer(-1)->getnDesign());
  }

  /* if set, overwrite classification design from file */
  if (strcmp(classificationfilename, "NONE") != 0) {
    sprintf(filename, "%s/%s", datafolder, classificationfilename);
    read_vector(filename, getLayer(nlayers_global-2)->getWeights(), getLayer(nlayers_global-2)->getnDesign());
  }

  /* Communicate the neighbours across processors */
  MPI_CommunicateNeighbours();
}


void Network::interpolateDesign(int rfactor, Network* coarse_net, int NI_interp_type){

  int nDim;
  Layer *clayer_left = NULL;
  Layer *clayer_right = NULL;
  Layer *flayer = NULL;
  int clayerID;

  /* Loop over all ayers */
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) { 

      /* Get pointer to the fine-grid layer */
      flayer = layers[getLocalID(ilayer)];

      /* Get pointer to the left-sided coarse point */
      if (ilayer == -1){
        clayerID = -1;
      } else if (ilayer == nlayers_global - 2) {
        clayerID = coarse_net->getnLayersGlobal() - 2;
      } else {
        clayerID = (int) ilayer / rfactor;  // This floors to closest smaller integer. 
      }
      clayer_left = coarse_net->getLayer(clayerID);   
      if (clayer_left == NULL) {
        printf("\n\n%d: ERROR: Can't access coarse-grid layer! flayerID = %d, clayerID = %d\n", mpirank, ilayer, clayerID );
        exit(1); // TODO: Fix it!
      }
     
      /* --- Interpolate the design and reset the gradient to 0 --- */
      nDim = clayer_left->getnDesign();

      /* Always copy clayer to flayer weights */
      vec_copy(nDim, clayer_left->getWeights(), flayer->getWeights());
      vec_setZero(nDim, flayer->getWeightsBar());

      if ( (NI_interp_type == 1)  &&  (ilayer != -1) && (ilayer != nlayers_global - 2 ) && (ilayer < nlayers_global - 2 - rfactor ) ) {
         /* If linear interpolation: Scale weights and add right clayer
          * weights, if not at classification or opening layer or at right-most hidden layer (where there is no layer to right to average with) */

        clayer_right = coarse_net->getLayer(clayerID + 1); 
        if (clayer_right == NULL) {
          printf("\n\n%d: ERROR: Can't access coarse-grid layer! flayerID = %d, clayerID = %d\n", mpirank, ilayer, clayerID );
          exit(1); // TODO: Fix it!
        }
        MyReal scale2 = ((MyReal) (ilayer % rfactor)) / ((MyReal) rfactor);
        MyReal scale1 = 1.0 - scale2;
        vec_scale(nDim, scale1, flayer->getWeights());
        vec_axpy(nDim, scale2, clayer_right->getWeights(), flayer->getWeights());
        //fprintf(stderr, "\nCheck   %d  %d  %d  %d  %d  %d\n", ilayer, clayerID+1, getLocalID(ilayer), startlayerID, endlayerID, nlayers_global);
      }
      else if ( (NI_interp_type == 2)  &&  (ilayer != -1) && (ilayer != nlayers_global - 2 ) ) {
         /* If 'Do Nothing' interpolation, set new layers to 0.  Then in main
          * increase t_final so that the initial network propagation is the same */
        
        if (ilayer % rfactor != 0){
          vec_setZero(nDim, flayer->getWeights());
        }
        //else do nothing
      }

  }

  /* Communicate ghost layers */
  MPI_CommunicateNeighbours();
}


int Network::MPI_CommunicateNeighbours() {

  MPI_Request sendlastreq, recvlastreq;
  MPI_Request sendfirstreq, recvfirstreq;
  MPI_Status status;
  Layer *layer = NULL;

  /* Allocate buffers */
  int size_left = -1;
  int size_right = -1;

  MyReal *sendlast = 0;
  MyReal *recvlast = 0;
  MyReal *sendfirst = 0;
  MyReal *recvfirst = 0;

  /* Don't communicate on processors that don't have layers */
  if (startlayerID > endlayerID) return 0; 

  /* --- All but the first proc receive the left neighbour's last layer --- */
  if (startlayerID > -1) {
    /* Receive from left neighbour */
    int source = mpirank - 1;

    size_left = layer_left->getnDesign();
    recvlast = new MyReal[size_left];

    MPI_Irecv(recvlast, size_left, MPI_MyReal, source, 0, comm, &recvlastreq);
  }

  /* --- All but the process with the classification layer sent their last layer to right neighbour --- */
  if (endlayerID < nlayers_global - 2) {
    layer = getLayer(endlayerID);
    size_left = layer->getnDesign();
    sendlast = new MyReal[size_left];

    /* Pack the last layer into a buffer */
    layer->packDesign(sendlast, size_left);

    /* Send to right neighbour */
    int receiver = mpirank + 1;
    MPI_Isend(sendlast, size_left, MPI_MyReal, receiver, 0, comm, &sendlastreq);
  }

  /* --- All but the process with the classification layer recv the right neighbour's first layer --- */
  if (endlayerID < nlayers_global - 2) {
    /* Receive from right neighbour */
    int source = mpirank + 1;

    size_right = layer_right->getnDesign();
    recvfirst = new MyReal[size_right];

    MPI_Irecv(recvfirst, size_right, MPI_MyReal, source, 1, comm, &recvfirstreq);
  }

  /* --- All but the first processor send their first layer to the left neighbour --- */
  if (startlayerID > -1) {
    layer = getLayer(startlayerID);
    size_right = layer->getnDesign();
    sendfirst = new MyReal[size_right];

    /* Pack the first layer into a buffer */
    layer->packDesign(sendfirst, size_right);

    /* Send to left neighbour */
    int receiver = mpirank - 1;
    MPI_Isend(sendfirst, size_right, MPI_MyReal, receiver, 1, comm, &sendfirstreq);
  }

  /* Wait to finish up communication */
  if (startlayerID > -1) MPI_Wait(&recvlastreq, &status);
  if (endlayerID < nlayers_global-2) MPI_Wait(&sendlastreq, &status);
  if (endlayerID < nlayers_global-2) MPI_Wait(&recvfirstreq, &status);
  if (startlayerID > -1) MPI_Wait(&sendfirstreq, &status);

  /* Unpack and store the left received layer */
  if (startlayerID > -1) {
    layer_left->unpackDesign(recvlast);
  }

  /* Unpack and store the right received layer */
  if (endlayerID < nlayers_global - 2) {
    layer_right->unpackDesign(recvfirst);
  }

  /* Free the buffer */
  if (sendlast != 0) delete[] sendlast;
  if (recvlast != 0) delete[] recvlast;
  if (sendfirst != 0) delete[] sendfirst;
  if (recvfirst != 0) delete[] recvfirst;

  return 0;
}

void Network::evalClassification(DataSet *data, MyReal **state, int output) {
  MyReal *tmpstate = new MyReal[nchannels];

  int class_id;
  int success, success_local;
  FILE *classfile;
  ClassificationLayer *classificationlayer;

  /* Get classification layer */
  classificationlayer =
      dynamic_cast<ClassificationLayer *>(getLayer(nlayers_global - 2));
  if (classificationlayer == NULL) {
    printf("\n ERROR: Network can't access classification layer!\n\n");
    exit(1);
  }

  /* open file for printing predicted file */
  if (output) classfile = fopen("classprediction.dat", "w");

  loss = 0.0;
  accuracy = 0.0;
  success = 0;
  for (int iex = 0; iex < data->getnBatch(); iex++) {
    /* Copy values so that they are not overwrittn (they are needed for
     * adjoint)*/
    for (int ic = 0; ic < nchannels; ic++) {
      tmpstate[ic] = state[iex][ic];
    }
    /* Apply classification on tmpstate */
    classificationlayer->setLabel(data->getLabel(iex));
    classificationlayer->applyFWD(tmpstate);
    /* Evaluate Loss */
    loss += classificationlayer->crossEntropy(tmpstate);
    success_local = classificationlayer->prediction(tmpstate, &class_id);
    success += success_local;
    if (output) fprintf(classfile, "%d   %d\n", class_id, success_local);
  }
  loss = 1. / data->getnBatch() * loss;
  accuracy = 100.0 * ((MyReal)success) / data->getnBatch();
  // printf("Classification %d: %1.14e using layer %1.14e state %1.14e
  // tmpstate[0] %1.14e\n", getIndex(), loss, weights[0], state[1][1],
  // tmpstate[0]);

  if (output) fclose(classfile);
  if (output) printf("Prediction file written: classprediction.dat\n");

  delete[] tmpstate;
}

void Network::evalClassification_diff(DataSet *data, MyReal **primalstate,
                                      MyReal **adjointstate,
                                      int compute_gradient) {
  MyReal *tmpstate = new MyReal[nchannels];
  ClassificationLayer *classificationlayer;

  /* Get classification layer */
  classificationlayer =
      dynamic_cast<ClassificationLayer *>(getLayer(nlayers_global - 2));
  if (classificationlayer == NULL) {
    printf("\n ERROR: Network can't access classification layer!\n\n");
    exit(1);
  }

  int nbatch = data->getnBatch();
  MyReal loss_bar = 1. / nbatch;

  for (int iex = 0; iex < nbatch; iex++) {
    /* Recompute the Classification */
    for (int ic = 0; ic < nchannels; ic++) {
      tmpstate[ic] = primalstate[iex][ic];
    }
    classificationlayer->setLabel(data->getLabel(iex));
    classificationlayer->applyFWD(tmpstate);

    /* Derivative of Loss and classification. */
    classificationlayer->crossEntropy_diff(tmpstate, adjointstate[iex],
                                           loss_bar);
    classificationlayer->applyBWD(primalstate[iex], adjointstate[iex],
                                  compute_gradient);
  }
  // printf("Classification_diff %d using layer %1.14e state %1.14e tmpstate
  // %1.14e biasbar[dimOut-1] %1.14e\n", getIndex(), weights[0],
  // primalstate[1][1], tmpstate[0], bias_bar[dim_Out-1]);

  delete[] tmpstate;

}


