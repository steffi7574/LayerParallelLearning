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
  openlayer = NULL;
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

  /* Create Opening layer on first processor */
  if (mpirank == 0) {
    int index = -1;
    openlayer = createLayer(index, config);
    ndesign_local += openlayer->getnDesign();
    // printf("Create opening layer %d, ndesign_local %d \n", index,
    // openlayer->getnDesign());
  }

  /* Create vector of layers on this processor (hidden and classification) */
  layers = new Layer *[nlayers_local];  
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) {
    /* Create a layer at time step ilayer. Local storage at ilayer - startlayerID */
    int storeID = getLocalID(ilayer);
    layers[storeID] = createLayer(ilayer, config);

    /* Update parameters */
    int ndesign_thislayer = layers[storeID]->getnDesign();
    ndesign_local += ndesign_thislayer;
    if (ndesign_thislayer > mylayermax && ilayer < nlayers_global - 2) 
      mylayermax = ndesign_thislayer;
    // printf("creating hidden/class layer %d/%d, ndesign_local%d\n", ilayer,
    // nlayers_local, layers[storeID]->getnDesign());
  }

  /* Allocate memory for network design and gradient variables */
  design = new MyReal[ndesign_local];
  gradient = new MyReal[ndesign_local];

  /* Set the memory locations for all layers */
  int istart = 0;
  if (openlayer != NULL)  // Openlayer on first processor
  {
    openlayer->setMemory(&(design[istart]), &(gradient[istart]));
    istart += openlayer->getnDesign();
  }
  for (int ilayer = startlayerID; ilayer <= endlayerID;
       ilayer++)  // intermediate and classification layers
  {
    layers[getLocalID(ilayer)]->setMemory(&(design[istart]),
                                          &(gradient[istart]));
    istart += layers[getLocalID(ilayer)]->getnDesign();
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
  /* Delete openlayer */
  if (openlayer != NULL) delete openlayer;

  /* Delete intermediate and classification layers */
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
  Layer *layer;

  if (layerindex == -1)  // opening layer
  {
    layer = openlayer;
  } else if (layerindex == startlayerID - 1) {
    layer = layer_left;
  } else if (startlayerID <= layerindex && layerindex <= endlayerID) {
    layer = layers[getLocalID(layerindex)];
  } else if (layerindex == endlayerID + 1) {
    layer = layer_right;
  } else {
    layer = NULL;
  }

  return layer;
}

int Network::getnDesignLayermax() { return ndesign_layermax; }


void Network::setDesignRandom(double factor_openweights, double factor_hiddenweights, double factor_classiweights ){
  MyReal factor;
  MyReal *design_init;
  int myid;
  MPI_Comm_rank(comm, &myid);

  /* Initialize design with random numbers (do on one processor and scatter for
   * scaling test) */
  if (myid == 0) {
    srand(1.0);
    design_init = new MyReal[ndesign_global];
    for (int i = 0; i < ndesign_global; i++) {
      design_init[i] = (MyReal)rand() / ((MyReal)RAND_MAX);
    }
  }
  /* Scatter initial design to all processors */
  MPI_ScatterVector(design_init, design, ndesign_local, 0, comm);

  /* Scale the opening layer weights on first processor and reset gradient */
  if (startlayerID == 0) {
    factor = factor_openweights;
    vec_scale(openlayer->getnDesign(), factor, openlayer->getWeights());
    vec_setZero(openlayer->getnDesign(), openlayer->getWeightsBar());
  }

  /* Scale the intermediate (hidden) and classification layers and reset gradient */
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) {
    if (ilayer < nlayers_global - 2)  // Intermediate layer
    {
      factor = factor_hiddenweights;
    } else  // Classification layer
    {
      factor = factor_classiweights;
    }
    int storeID = getLocalID(ilayer);
    vec_scale(layers[storeID]->getnDesign(), factor, layers[storeID]->getWeights());
    vec_setZero(layers[storeID]->getnDesign(), layers[storeID]->getWeightsBar());

  }

  /* Communicate the neighbours across processors */
  MPI_CommunicateNeighbours();

  if (myid == 0) delete[] design_init;
}


void Network::interpolateDesign(int rfactor, Network* coarse_net, int NI_interp_type){

  int nDim;
  Layer *clayer_left, *clayer_right, *flayer;

  /* Copy the opening layer, which is stored on the first processor */
  if (mpirank == 0){
    clayer_left = coarse_net->openlayer;
    flayer = openlayer;
    nDim = openlayer->getnDesign();
    vec_copy(nDim, clayer_left->getWeights(), flayer->getWeights());
    vec_setZero(nDim, flayer->getWeightsBar());
  }

  /* Interpolate hidden layers (only copying so far) */
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) { 
    if (ilayer == nlayers_global-2) continue; // this excludes the classification layer

      /* Get pointer to the left-sided coarse point */
      int clayerID = ilayer / rfactor;  // This should round down, floor(ilayer/rfactor) !
      clayer_left = coarse_net->getLayer(clayerID);   
      if (clayer_left == NULL) {
        printf("\n\n ERROR: Can't access left neighbouring coarse-grid layer!\n");
        printf("\n\n Might be that it's not stored on this processor. Need to communicate? To be implemented...\n");
        // TODO: Fix it!
        exit(1);
      }

      /* If doing linear, get pointer to the right-sided coarse point (if not at last hidden layer) */
      if ( (NI_interp_type == 1)  &&  ((clayerID+1) != (coarse_net->getnLayersGlobal()-2)) ) {
        clayer_right = coarse_net->getLayer(clayerID + 1); // Point to the right neighbor   
        if (clayer_right == NULL) {
          printf("\n\n ERROR: Can't access right neighbouring coarse-grid layer!\n");
          printf("\n\n Might be that it's not stored on this processor. Need to communicate? To be implemented...\n");
          // TODO: Fix it!
          exit(1);
          }
      }

      
      /* Get pointer to the fine-grid layer */
      flayer = getLayer(ilayer);

      /* Interpolate the design and reset the gradient to 0
       * 
       * Note: Always do piece-wise constant interpolation for NI_interp_type 0,
       *       OR if you are in the last interval of new layers being added   */
      nDim = clayer_left->getnDesign();
      vec_copy(nDim, clayer_left->getWeights(), flayer->getWeights());
      vec_setZero(nDim, flayer->getWeightsBar());
      if ( (NI_interp_type == 1)  &&  ((clayerID+1) != (coarse_net->getnLayersGlobal()-2)) ) {
        /* Do linear interp because we are not at last interval of layers and NI_interp_type is 1 */
        MyReal scale2 = ((MyReal) (ilayer % rfactor)) / ((MyReal) rfactor);
        MyReal scale1 = 1.0 - scale2;
        vec_scale(nDim, scale1, flayer->getWeights());
        vec_axpy(nDim, scale2, clayer_right->getWeights(), flayer->getWeights());
        //printf("\nCheck   %d  %f  %f \n",ilayer, scale1, scale2);
      }
  }

  /* Copy classification layer, which is stored on the last processor */
  clayer_left = coarse_net->getLayer(coarse_net->getnLayersGlobal()-2); 
  if (clayer_left != NULL)
  {
    flayer = getLayer(nlayers_global - 2);
    nDim =  clayer_left->getnDesign();
    vec_copy(nDim, clayer_left->getWeights(), flayer->getWeights());
    vec_setZero(nDim, flayer->getWeightsBar());
  }
  /* Communicate ghost layers */
  MPI_CommunicateNeighbours();
}


void Network::setDesignFromFile(const char* datafolder, const char* openlayerfile, const char* hiddenlayerfile, const char* classificationlayerfile) {
  char filename[255];
  bool communicate = false;

  /* Read the opening weights, if set (on first processor only) */
  if (startlayerID == 0) {
    if (strcmp(openlayerfile, "NONE") != 0) {
      sprintf(filename, "%s/%s", datafolder, openlayerfile);
      read_vector(filename, openlayer->getWeights(), openlayer->getnDesign());
      vec_setZero(openlayer->getnDesign(), openlayer->getWeightsBar());
      communicate = true;
    }
  }

  /* Read the classification layer weights, if set (on last processor only) */
  if (endlayerID == nlayers_global - 2) {
    int storeID = getLocalID(nlayers_global-2);
    if (strcmp(classificationlayerfile, "NONE") != 0) {
      sprintf(filename, "%s/%s", datafolder, classificationlayerfile);
      read_vector(filename, layers[storeID]->getWeights(), layers[storeID]->getnDesign());
      vec_setZero(layers[storeID]->getnDesign(), layers[storeID]->getWeightsBar());
      communicate = true;
    }
  }

  /* Communicate the neighbours across processors */
  if (communicate) MPI_CommunicateNeighbours();
}

void Network::MPI_CommunicateNeighbours() {
  int myid, comm_size;
  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &comm_size);
  MPI_Request sendlastreq, recvlastreq;
  MPI_Request sendfirstreq, recvfirstreq;
  MPI_Status status;

  /* Allocate buffers */
  int size_left = -1;
  int size_right = -1;

  MyReal *sendlast = 0;
  MyReal *recvlast = 0;
  MyReal *sendfirst = 0;
  MyReal *recvfirst = 0;

  /* --- All but the first process receive the last layer from left neighbour
   * --- */
  if (myid > 0) {
    /* Receive from left neighbour */
    int source = myid - 1;

    size_left = layer_left->getnDesign();
    recvlast = new MyReal[size_left];

    MPI_Irecv(recvlast, size_left, MPI_MyReal, source, 0, comm, &recvlastreq);
  }

  /* --- All but the last process sent their last layer to right neighbour ---
   */
  if (myid < comm_size - 1) {
    size_left = layers[getLocalID(endlayerID)]->getnDesign();
    sendlast = new MyReal[size_left];

    /* Pack the last layer into a buffer */
    layers[getLocalID(endlayerID)]->packDesign(sendlast, size_left);

    /* Send to right neighbour */
    int receiver = myid + 1;
    MPI_Isend(sendlast, size_left, MPI_MyReal, receiver, 0, comm, &sendlastreq);
  }

  /* --- All but the last processor recv the first layer from the right
   * neighbour --- */
  if (myid < comm_size - 1) {
    /* Receive from right neighbour */
    int source = myid + 1;

    size_right = layer_right->getnDesign();
    recvfirst = new MyReal[size_right];

    MPI_Irecv(recvfirst, size_right, MPI_MyReal, source, 1, comm,
              &recvfirstreq);
  }

  /* --- All but the first processor send their first layer to the left
   * neighbour --- */
  if (myid > 0) {
    size_right = layers[getLocalID(startlayerID)]->getnDesign();
    sendfirst = new MyReal[size_right];

    /* Pack the first layer into a buffer */
    layers[getLocalID(startlayerID)]->packDesign(sendfirst, size_right);

    /* Send to left neighbour */
    int receiver = myid - 1;
    MPI_Isend(sendfirst, size_right, MPI_MyReal, receiver, 1, comm,
              &sendfirstreq);
  }

  /* Wait to finish up communication */
  if (myid > 0) MPI_Wait(&recvlastreq, &status);
  if (myid < comm_size - 1) MPI_Wait(&sendlastreq, &status);
  if (myid < comm_size - 1) MPI_Wait(&recvfirstreq, &status);
  if (myid > 0) MPI_Wait(&sendfirstreq, &status);

  /* Unpack and store the left received layer */
  if (myid > 0) {
    layer_left->unpackDesign(recvlast);
  }

  /* Unpack and store the right received layer */
  if (myid < comm_size - 1) {
    layer_right->unpackDesign(recvfirst);
  }

  /* Free the buffer */
  if (sendlast != 0) delete[] sendlast;
  if (recvlast != 0) delete[] recvlast;
  if (sendfirst != 0) delete[] sendfirst;
  if (recvfirst != 0) delete[] recvfirst;
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


