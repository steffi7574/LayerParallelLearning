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
#include "braid_wrapper.hpp"

/* ========================================================= */
myBraidVector::myBraidVector(int nChannels, int nBatch) {
  nchannels = nChannels;
  nbatch = nBatch;

  state = NULL;
  layer = NULL;
  sendflag = -1.0;

  /* Allocate the state vector */
  state = new MyReal *[nbatch];
  for (int iex = 0; iex < nbatch; iex++) {
    state[iex] = new MyReal[nchannels];
    for (int ic = 0; ic < nchannels; ic++) {
      state[iex][ic] = 0.0;
    }
  }
}

myBraidVector::~myBraidVector() {
  /* Deallocate the state vector */
  for (int iex = 0; iex < nbatch; iex++) {
    delete[] state[iex];
  }
  delete[] state;
  state = NULL;
}

int myBraidVector::getnChannels() { return nchannels; }

int myBraidVector::getnBatch() { return nbatch; }

MyReal *myBraidVector::getState(int exampleID) { return state[exampleID]; }

MyReal **myBraidVector::getState() { return state; }

Layer *myBraidVector::getLayer() { return layer; }
void myBraidVector::setLayer(Layer *layerptr) { layer = layerptr; }

MyReal myBraidVector::getSendflag() { return sendflag; }
void myBraidVector::setSendflag(MyReal value) { sendflag = value; }

/* ========================================================= */
/* ========================================================= */
/* ========================================================= */
myBraidApp::myBraidApp(DataSet *Data, Network *Network, Config *config,
                       MPI_Comm comm, int current_nlayers)
    : BraidApp(comm, 0.0, config->T, current_nlayers) {
  MPI_Comm_rank(comm, &myid);
  network = Network;
  data = Data;
  objective = 0.0;

  /* Allocate and Initialize (braid_Init) XBraid core */
  core = new BraidCore(comm, this);

  /* Set braid options */
  core->SetStorage(0);  // TODO Do we want this SetStorage here? 
  core->SetMaxLevels(config->braid_maxlevels);
  core->SetMinCoarse(config->braid_mincoarse);
  core->SetPrintLevel(config->braid_printlevel);
  core->SetCFactor(0, config->braid_cfactor0);
  core->SetCFactor(-1, config->braid_cfactor);
  core->SetAccessLevel(config->braid_accesslevel);
  core->SetMaxIter(config->braid_maxiter);
  core->SetSkip(config->braid_setskip);
  if (config->braid_fmg) {
    core->SetFMG();
  }
  core->SetNRelax(-1, config->braid_nrelax);
  core->SetNRelax(0, config->braid_nrelax0);
  core->SetAbsTol(config->braid_abstol);
}

myBraidApp::~myBraidApp() {
  /* Delete the core, if drive() has been called */
  if (core->GetWarmRestart()) delete core;
}

MyReal myBraidApp::getObjective() { return objective; }

BraidCore *myBraidApp::getCore() { return core; }

void myBraidApp::Refine(int rfactor, Network *Network, DataSet *Data,  int current_nlayers)
{
  /* Use FRefine to refine the grid, i.e., add new layers */
    
  /* Set the refinement factors in core to rfactor */
  braid_Int *rfactors = _braid_CoreElt(core->GetCore(), rfactors);
  _braid_Grid **grids = _braid_CoreElt(core->GetCore(), grids);
  int ilower  = _braid_GridElt(grids[0], ilower);
  int iupper  = _braid_GridElt(grids[0], iupper);
  for( int i = ilower; i <= iupper; i++) {
    rfactors[i - ilower] = rfactor;
  }

  /* Call FRefine to create new core with more layers */
  int refined = 0;
  core->SetRefine(1);
  core->SetStorage(0);
  _braid_FRefine(core->GetCore(), &refined);
  core->SetRefine(0);
  if(refined == 0) {
     printf("\n Error: FRefine() failed.\n");
  }

  /* Update network weights, data */
  network = Network;
  data = Data;
  
  /* Update to new number of layers */
  ntime = current_nlayers;
}

void myBraidApp::GetGridDistribution(int *ilower_ptr, int *iupper_ptr) {
  core->GetDistribution(ilower_ptr, iupper_ptr);
}

braid_Int myBraidApp::GetTimeStepIndex(MyReal t) {
  /* Round to the closes integer */
  int ts = round(t / network->getDT());
  return ts;
}

braid_Int myBraidApp::Step(braid_Vector u_, braid_Vector ustop_,
                           braid_Vector fstop_, BraidStepStatus &pstatus) {
  int ts_stop;
  MyReal tstart, tstop;
  MyReal deltaT;

  myBraidVector *u = (myBraidVector *)u_;
  int nbatch = data->getnBatch();

  /* Get the time-step size and current time index*/
  pstatus.GetTstartTstop(&tstart, &tstop);
  ts_stop = GetTimeStepIndex(tstop);
  deltaT = tstop - tstart;

  /* Set time step size */
  u->getLayer()->setDt(deltaT);

  //printf("DT  %1.16e \n", tstop - tstart);
  // printf("%d: step %d,%f -> %d, %f layer %d using %1.14e state %1.14e, %d\n",
  // app->myid, tstart, ts_stop, tstop, u->layer->getIndex(),
  // u->layer->getWeights()[3], u->state[1][1], u->layer->getnDesign());

  /* apply the layer for all examples */
  for (int iex = 0; iex < nbatch; iex++) {
    /* Apply the layer */
    u->getLayer()->applyFWD(u->getState(iex));
  }

  /* Free the layer, if it has just been send to this processor */
  if (u->getSendflag() > 0.0) {
    delete[] u->getLayer()->getWeights();
    delete[] u->getLayer()->getWeightsBar();
    delete u->getLayer();
  }
  u->setSendflag(-1.0);

  /* Move the layer pointer of u forward to that of tstop */
  u->setLayer(network->getLayer(ts_stop));

  /* no refinement */
  pstatus.SetRFactor(1);

  return 0;
}

/* Compute residual: Does nothing. */
braid_Int myBraidApp::Residual(braid_Vector u_, braid_Vector r_,
                               BraidStepStatus &pstatus) {
  printf("\n\n I SHOUD NOT BE CALLED... I AM NOT IMPLEMENTED!\n\n");

  return 0;
}

braid_Int myBraidApp::Clone(braid_Vector u_, braid_Vector *v_ptr) {
  myBraidVector *u = (myBraidVector *)u_;

  int nchannels = u->getnChannels();
  int nbatch = u->getnBatch();

  /* Allocate a new vector */
  myBraidVector *v = new myBraidVector(nchannels, nbatch);

  /* Copy the values */
  for (int iex = 0; iex < nbatch; iex++) {
    for (int ic = 0; ic < nchannels; ic++) {
      v->getState(iex)[ic] = u->getState(iex)[ic];
    }
  }
  v->setLayer(u->getLayer());
  v->setSendflag(u->getSendflag());

  /* Set the return pointer */
  *v_ptr = (braid_Vector)v;

  return 0;
}

braid_Int myBraidApp::Init(braid_Real t, braid_Vector *u_ptr) {
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();

  myBraidVector *u = new myBraidVector(nchannels, nbatch);

  /* Apply the opening layer */
  if (t == 0) {
    Layer *openlayer = network->getLayer(-1);
    // printf("%d: Init %f: layer %d using %1.14e state %1.14e, %d\n",
    // app->myid, t, openlayer->getIndex(), openlayer->getWeights()[3],
    // u->state[1][1], openlayer->getnDesign());
    for (int iex = 0; iex < nbatch; iex++) {
      /* set example */
      openlayer->setExample(data->getExample(iex));

      /* Apply the layer */
      openlayer->applyFWD(u->getState(iex));
    }
  }

  /* Set the layer pointer */
  if (t >= 0)  // this should always be the case...
  {
    int ilayer = GetTimeStepIndex(t);
    u->setLayer(network->getLayer(ilayer));
  }

  /* Return the pointer */
  *u_ptr = (braid_Vector)u;

  return 0;
}

braid_Int myBraidApp::Free(braid_Vector u_) {
  myBraidVector *u = (myBraidVector *)u_;
  delete u;
  return 0;
}

braid_Int myBraidApp::Sum(braid_Real alpha, braid_Vector x_, braid_Real beta,
                          braid_Vector y_) {
  myBraidVector *x = (myBraidVector *)x_;
  myBraidVector *y = (myBraidVector *)y_;

  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();

  for (int iex = 0; iex < nbatch; iex++) {
    for (int ic = 0; ic < nchannels; ic++) {
      y->getState(iex)[ic] =
          alpha * (x->getState(iex)[ic]) + beta * (y->getState(iex)[ic]);
    }
  }

  return 0;
}

braid_Int myBraidApp::SpatialNorm(braid_Vector u_, braid_Real *norm_ptr) {
  myBraidVector *u = (myBraidVector *)u_;
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();

  MyReal dot = 0.0;
  for (int iex = 0; iex < nbatch; iex++) {
    dot += vecdot(nchannels, u->getState(iex), u->getState(iex));
  }
  *norm_ptr = sqrt(dot) / nbatch;

  return 0;
}

braid_Int myBraidApp::Access(braid_Vector u_, BraidAccessStatus &astatus) {
  printf("my_Access: To be implemented...\n");

  return 0;
}

braid_Int myBraidApp::BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus) {
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();

  /* Gather number of variables */
  int nuvector = nchannels * nbatch;
  int nlayerinfo = 12;
  int nlayerdesign = network->getnDesignLayermax();

  /* Set the size */
  *size_ptr = (nuvector + nlayerinfo + nlayerdesign) * sizeof(MyReal);

  return 0;
}

braid_Int myBraidApp::BufPack(braid_Vector u_, void *buffer,
                              BraidBufferStatus &bstatus) {
  int size;
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();
  MyReal *dbuffer = (MyReal *)buffer;
  myBraidVector *u = (myBraidVector *)u_;

  /* Store network state */
  int idx = 0;
  for (int iex = 0; iex < nbatch; iex++) {
    for (int ic = 0; ic < nchannels; ic++) {
      dbuffer[idx] = u->getState(iex)[ic];
      idx++;
    }
  }
  size = nchannels * nbatch * sizeof(MyReal);

  int nweights = u->getLayer()->getnWeights();
  int nbias = u->getLayer()->getDimBias();

  dbuffer[idx] = u->getLayer()->getType();
  idx++;
  dbuffer[idx] = u->getLayer()->getIndex();
  idx++;
  dbuffer[idx] = u->getLayer()->getDimIn();
  idx++;
  dbuffer[idx] = u->getLayer()->getDimOut();
  idx++;
  dbuffer[idx] = u->getLayer()->getDimBias();
  idx++;
  dbuffer[idx] = u->getLayer()->getnWeights();
  idx++;
  dbuffer[idx] = u->getLayer()->getActivation();
  idx++;
  dbuffer[idx] = u->getLayer()->getnDesign();
  idx++;
  dbuffer[idx] = u->getLayer()->getGammaTik();
  idx++;
  dbuffer[idx] = u->getLayer()->getGammaDDT();
  idx++;
  dbuffer[idx] = u->getLayer()->getnConv();
  idx++;
  dbuffer[idx] = u->getLayer()->getCSize();
  idx++;
  for (int i = 0; i < nweights; i++) {
    dbuffer[idx] = u->getLayer()->getWeights()[i];
    idx++;
    // dbuffer[idx] = u->layer->getWeightsBar()[i];  idx++;
  }
  for (int i = 0; i < nbias; i++) {
    dbuffer[idx] = u->getLayer()->getBias()[i];
    idx++;
    // dbuffer[idx] = u->layer->getBiasBar()[i];  idx++;
  }
  size += (12 + (nweights + nbias)) * sizeof(MyReal);

  bstatus.SetSize(size);

  return 0;
}

braid_Int myBraidApp::BufUnpack(void *buffer, braid_Vector *u_ptr,
                                BraidBufferStatus &bstatus) {
  Layer *tmplayer = 0;
  MyReal *dbuffer = (MyReal *)buffer;

  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();

  /* Allocate a new vector */
  myBraidVector *u = new myBraidVector(nchannels, nbatch);

  /* Unpack the buffer */
  int idx = 0;
  for (int iex = 0; iex < nbatch; iex++) {
    for (int ic = 0; ic < nchannels; ic++) {
      u->getState(iex)[ic] = dbuffer[idx];
      idx++;
    }
  }

  /* Receive and initialize a layer. Set the sendflag */
  int layertype = dbuffer[idx];
  idx++;
  int index = dbuffer[idx];
  idx++;
  int dimIn = dbuffer[idx];
  idx++;
  int dimOut = dbuffer[idx];
  idx++;
  int dimBias = dbuffer[idx];
  idx++;
  int nweights = dbuffer[idx];
  idx++;
  int activ = dbuffer[idx];
  idx++;
  int nDesign = dbuffer[idx];
  idx++;
  int gammatik = dbuffer[idx];
  idx++;
  int gammaddt = dbuffer[idx];
  idx++;
  int nconv = dbuffer[idx];
  idx++;
  int csize = dbuffer[idx];
  idx++;

  /* layertype decides on which layer should be created */
  switch (layertype) {
    case Layer::OPENZERO:
      tmplayer = new OpenExpandZero(dimIn, dimOut);
      break;
    case Layer::OPENDENSE:
      tmplayer = new OpenDenseLayer(dimIn, dimOut, activ, gammatik);
      break;
    case Layer::DENSE:
      tmplayer =
          new DenseLayer(index, dimIn, dimOut, 1.0, activ, gammatik, gammaddt);
      break;
    case Layer::CLASSIFICATION:
      tmplayer = new ClassificationLayer(index, dimIn, dimOut, gammatik);
      break;
    case Layer::OPENCONV:
      tmplayer = new OpenConvLayer(dimIn, dimOut);
      break;
    case Layer::OPENCONVMNIST:
      tmplayer = new OpenConvLayerMNIST(dimIn, dimOut);
      break;
    case Layer::CONVOLUTION:
      tmplayer = new ConvLayer(index, dimIn, dimOut, csize, nconv, 1.0, activ,
                               gammatik, gammaddt);
      break;
    default:
      printf("\n\n ERROR while unpacking a buffer: Layertype unknown!!\n\n");
  }

  /* Allocate design and gradient */
  MyReal *design = new MyReal[nDesign];
  MyReal *gradient = new MyReal[nDesign];
  tmplayer->setMemory(design, gradient);
  /* Set the weights */
  for (int i = 0; i < nweights; i++) {
    tmplayer->getWeights()[i] = dbuffer[idx];
    idx++;
  }
  for (int i = 0; i < dimBias; i++) {
    tmplayer->getBias()[i] = dbuffer[idx];
    idx++;
  }
  u->setLayer(tmplayer);
  u->setSendflag(1.0);

  /* Return the pointer */
  *u_ptr = (braid_Vector)u;
  return 0;
}

braid_Int myBraidApp::SetInitialCondition() {
  Layer *openlayer = network->getLayer(-1);
  int nbatch = data->getnBatch();
  braid_BaseVector ubase;
  myBraidVector *u;

  /* Apply initial condition if warm_restart (otherwise it is set in my_Init()
   */
  /* can not be set here if !(warm_restart) because braid_grid is created only
   * in braid_drive(). */
  if (core->GetWarmRestart()) {
    /* Get vector at t == 0 */
    _braid_UGetVectorRef(core->GetCore(), 0, 0, &ubase);
    if (ubase != NULL)  // only true on one first processor !
    {
      u = (myBraidVector *)ubase->userVector;

      /* Apply opening layer */
      for (int iex = 0; iex < nbatch; iex++) {
        /* set example */
        openlayer->setExample(data->getExample(iex));

        /* Apply the layer */
        openlayer->applyFWD(u->getState(iex));
      }
    }
  }

  return 0;
}

braid_Int myBraidApp::EvaluateObjective() {
  braid_BaseVector ubase;
  myBraidVector *u;
  Layer *layer;
  MyReal myobjective;
  MyReal regul;

  /* Get range of locally stored layers */
  int startlayerID = network->getStartLayerID();
  int endlayerID = network->getEndLayerID();
  if (startlayerID == 0)
    startlayerID -=
        1;  // this includes opening layer (id = -1) at first processor

  /* Iterate over the local layers */
  regul = 0.0;
  for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++) {
    /* Get the layer */
    layer = network->getLayer(ilayer);

    /* Tikhonov - Regularization*/
    regul += layer->evalTikh();

    /* DDT - Regularization on intermediate layers */
    regul +=
        layer->evalRegulDDT(network->getLayer(ilayer - 1), network->getDT());

    /* At last layer: Classification and Loss evaluation */
    if (ilayer == network->getnLayersGlobal() - 2) {
      _braid_UGetLast(core->GetCore(), &ubase);
      u = (myBraidVector *)ubase->userVector;
      network->evalClassification(data, u->getState(), 0);
    }
    // printf("%d: layerid %d using %1.14e, tik %1.14e, ddt %1.14e, loss
    // %1.14e\n", app->myid, layer->getIndex(), layer->getWeights()[0],
    // regultik, regulddt, loss_loc);
  }

  /* Collect objective function from all processors */
  myobjective = network->getLoss() + regul;
  objective = 0.0;
  MPI_Allreduce(&myobjective, &objective, 1, MPI_MyReal, MPI_SUM,
                MPI_COMM_WORLD);

  return 0;
}

MyReal myBraidApp::run() {
  int nreq = -1;
  MyReal norm;

  SetInitialCondition();
  core->Drive();
  EvaluateObjective();
  core->GetRNorms(&nreq, &norm);

  return norm;
}

/* ========================================================= */
/* ========================================================= */
/* ========================================================= */
myAdjointBraidApp::myAdjointBraidApp(DataSet *Data, Network *Network,
                                     Config *config, BraidCore *Primalcoreptr,
                                     MPI_Comm comm, int current_nlayers)
    : myBraidApp(Data, Network, config, comm, current_nlayers) {
  primalcore = Primalcoreptr;

  /* Store all primal points */
  primalcore->SetStorage(0);

  /* Revert processor ranks for solving adjoint with xbraid */
  core->SetRevertedRanks(1);
}

myAdjointBraidApp::~myAdjointBraidApp() {}


void myAdjointBraidApp::SetPrimalCore(BraidCore *Primalcoreptr)
{
  /* Update core structures */
  primalcore =  Primalcoreptr;
  primalcore->SetStorage(0);
}

void myAdjointBraidApp::Refine(int rfactor, Network *Network, DataSet *Data,  int current_nlayers, BraidCore *Primalcoreptr)
{
  /* Update core structures */
  //primalcore =  Primalcoreptr;
  //primalcore->SetStorage(0);

  /* Use FRefine to refine the grid, i.e., add new layers */
    
  /* Set the refinement factors in core to rfactor */
  braid_Int *rfactors = _braid_CoreElt(core->GetCore(), rfactors);
  _braid_Grid **grids = _braid_CoreElt(core->GetCore(), grids);
  int ilower  = _braid_GridElt(grids[0], ilower);
  int iupper  = _braid_GridElt(grids[0], iupper);
  for( int i = ilower; i <= iupper; i++) {
    rfactors[i - ilower] = rfactor;
  }

  /* Call FRefine to create new core with more layers */
  int refined = 0;
  core->SetRefine(1);
  _braid_FRefine(core->GetCore(), &refined);
  core->SetRefine(0);
  if(refined == 0) {
     printf("\n Error: FRefine() failed.\n");
  }
  
  /* Update network weights, data */
  network = Network;
  data = Data;
  
  /* Update to new number of layers */
  ntime = current_nlayers;
  
  /* Set Reverted Ranks for Backwards in Time */
  core->SetRevertedRanks(1);

}


int myAdjointBraidApp::GetPrimalIndex(int ts) {
  int idx = network->getnLayersGlobal() - 2 - ts;
  return idx;
}

braid_Int myAdjointBraidApp::Step(braid_Vector u_, braid_Vector ustop_,
                                  braid_Vector fstop_,
                                  BraidStepStatus &pstatus) {
  int ts_stop;
  int level, compute_gradient;
  MyReal tstart, tstop;
  MyReal deltaT;
  int finegrid = 0;
  int primaltimestep;
  braid_BaseVector ubaseprimal;
  myBraidVector *uprimal;

  int nbatch = data->getnBatch();
  myBraidVector *u = (myBraidVector *)u_;

  /* Update gradient only on the finest grid */
  pstatus.GetLevel(&level);
  if (level == 0)
    compute_gradient = 1;
  else
    compute_gradient = 0;

  /* Get the time-step size and current time index*/
  pstatus.GetTstartTstop(&tstart, &tstop);
  ts_stop = GetTimeStepIndex(tstop);
  deltaT = tstop - tstart;
  primaltimestep = GetPrimalIndex(ts_stop);

  /* Get the primal vector from the primal core */
  _braid_UGetVectorRef(primalcore->GetCore(), finegrid, primaltimestep,
                       &ubaseprimal);
  uprimal = (myBraidVector *)ubaseprimal->userVector;

  /* Reset gradient before the update */
  if (compute_gradient) 
      vec_setZero(uprimal->getLayer()->getnDesign(), uprimal->getLayer()->getWeightsBar());

  /* Take one step backwards, updates adjoint state and gradient, if desired. */
  uprimal->getLayer()->setDt(deltaT);
  for (int iex = 0; iex < nbatch; iex++) {
    uprimal->getLayer()->applyBWD(uprimal->getState(iex), u->getState(iex),
                                  compute_gradient);
  }

  // printf("%d: level %d step_adj %d->%d using layer %d,%1.14e, primal %1.14e,
  // adj %1.14e, grad[0] %1.14e, %d\n", app->myid, level, ts_stop,
  // uprimal->layer->getIndex(), uprimal->layer->getWeights()[3],
  // uprimal->state[1][1], u->state[1][1], uprimal->layer->getWeightsBar()[0],
  // uprimal->layer->getnDesign());

  /* Derivative of DDT-Regularization */
  if (compute_gradient) {
    Layer *prev = network->getLayer(primaltimestep - 1);
    Layer *next = network->getLayer(primaltimestep + 1);
    uprimal->getLayer()->evalRegulDDT_diff(prev, next, network->getDT());
  }

  /* Derivative of tikhonov */
  if (compute_gradient) uprimal->getLayer()->evalTikh_diff(1.0);

  /* no refinement */
  pstatus.SetRFactor(1);

  return 0;
}

braid_Int myAdjointBraidApp::Init(braid_Real t, braid_Vector *u_ptr) {
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();

  braid_BaseVector ubaseprimal;
  myBraidVector *uprimal;

  // printf("%d: Init %d (primaltimestep %d)\n", app->myid, ilayer,
  // primaltimestep);

  /* Allocate the adjoint vector and set to zero */
  myBraidVector *u = new myBraidVector(nchannels, nbatch);

  /* Adjoint initial (i.e. terminal) condition is derivative of classification
   * layer */
  if (t == 0) {
    /* Get the primal vector */
    _braid_UGetLast(primalcore->GetCore(), &ubaseprimal);
    if (ubaseprimal == NULL) return 0;
    
    uprimal = (myBraidVector *)ubaseprimal->userVector;

    /* Reset the gradient before updating it */
    vec_setZero(uprimal->getLayer()->getnDesign(), uprimal->getLayer()->getWeightsBar());

    /* Derivative of classification */
    network->evalClassification_diff(data, uprimal->getState(), u->getState(),
                                     1);

    /* Derivative of tikhonov regularization) */
    uprimal->getLayer()->evalTikh_diff(1.0);

    //    printf("%d: Init_adj Loss at %d, using %1.14e, primal %1.14e, adj
    //    %1.14e, grad[0] %1.14e\n", app->myid, layer->getIndex(),
    //    layer->getWeights()[0], primalstate[1][1], u->state[1][1],
    //    layer->getWeightsBar()[0]);
  }

  *u_ptr = (braid_Vector)u;

  return 0;
}

braid_Int myAdjointBraidApp::BufSize(braid_Int *size_ptr,
                                     BraidBufferStatus &bstatus) {
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();

  *size_ptr = nchannels * nbatch * sizeof(MyReal);
  return 0;
}

braid_Int myAdjointBraidApp::BufPack(braid_Vector u_, void *buffer,
                                     BraidBufferStatus &bstatus) {
  int size;
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();
  MyReal *dbuffer = (MyReal *)buffer;
  myBraidVector *u = (myBraidVector *)u_;

  /* Store network state */
  int idx = 0;
  for (int iex = 0; iex < nbatch; iex++) {
    for (int ic = 0; ic < nchannels; ic++) {
      dbuffer[idx] = u->getState(iex)[ic];
      idx++;
    }
  }
  size = nchannels * nbatch * sizeof(MyReal);

  bstatus.SetSize(size);
  return 0;
}

braid_Int myAdjointBraidApp::BufUnpack(void *buffer, braid_Vector *u_ptr,
                                       BraidBufferStatus &bstatus) {
  int nchannels = network->getnChannels();
  int nbatch = data->getnBatch();
  MyReal *dbuffer = (MyReal *)buffer;

  /* Allocate the vector */
  myBraidVector *u = new myBraidVector(nchannels, nbatch);

  /* Unpack the buffer */
  int idx = 0;
  for (int iex = 0; iex < nbatch; iex++) {
    for (int ic = 0; ic < nchannels; ic++) {
      u->getState(iex)[ic] = dbuffer[idx];
      idx++;
    }
  }
  u->setLayer(NULL);
  u->setSendflag(-1.0);

  *u_ptr = (braid_Vector)u;
  return 0;
}

braid_Int myAdjointBraidApp::SetInitialCondition() {
  braid_BaseVector ubaseprimal, ubaseadjoint;
  // braid_Vector      uprimal, uadjoint;
  myBraidVector *uprimal, *uadjoint;

  /* Only gradient for primal time step N here. Other time steps are in
   * my_Step_adj. */

  /* If warm_restart: set adjoint initial condition here. Otherwise it's set in
   * my_Init_Adj */
  /* It can not be done here if drive() has not been called before, because the
   * braid grid is allocated only at the beginning of drive() */
  if (core->GetWarmRestart()) {
    /* Get primal and adjoint state */
    _braid_UGetLast(primalcore->GetCore(), &ubaseprimal);
    _braid_UGetVectorRef(core->GetCore(), 0, 0, &ubaseadjoint);

    if (ubaseprimal != NULL &&
        ubaseadjoint != NULL)  // this is the case at first primal and last
                               // adjoint time step
    {
      uprimal = (myBraidVector *)ubaseprimal->userVector;
      uadjoint = (myBraidVector *)ubaseadjoint->userVector;

      /* Reset the gradient before updating it */
      vec_setZero(uprimal->getLayer()->getnDesign(), uprimal->getLayer()->getWeightsBar());

      // printf("%d: objective_diff at ilayer %d using %1.14e primal %1.14e\n",
      // app->myid, uprimal->layer->getIndex(), uprimal->layer->getWeights()[0],
      // uprimal->state[1][1]);

      /* Derivative of classification */
      network->evalClassification_diff(data, uprimal->getState(),
                                       uadjoint->getState(), 1);

      /* Derivative of tikhonov regularization) */
      uprimal->getLayer()->evalTikh_diff(1.0);
    }
  }

  return 0;
}

braid_Int myAdjointBraidApp::EvaluateObjective() {
  braid_BaseVector ubase;
  myBraidVector *uadjoint;

  Layer *openlayer = network->getLayer(-1);
  int nbatch = data->getnBatch();

  /* Get \bar y^0 (which is the LAST xbraid vector, stored on proc 0) */
  _braid_UGetLast(core->GetCore(), &ubase);
  if (ubase != NULL)  // This is true only on first processor (reverted ranks!)
  {
    uadjoint = (myBraidVector *)ubase->userVector;

    /* Reset the gradient */
    vec_setZero(openlayer->getnDesign(), openlayer->getWeightsBar());

    /* Apply opening layer backwards for all examples */
    for (int iex = 0; iex < nbatch; iex++) {
      openlayer->setExample(data->getExample(iex));
      /* TODO: Don't feed applyBWD with NULL! */
      openlayer->applyBWD(NULL, uadjoint->getState(iex), 1);
    }

    // printf("%d: Init_diff layerid %d using %1.14e, adj %1.14e grad[0]
    // %1.14e\n", app->myid, openlayer->getIndex(), openlayer->getWeights()[3],
    // ubase->userVector->state[1][1], openlayer->getWeightsBar()[0] );

    /* Derivative of Tikhonov Regularization */
    openlayer->evalTikh_diff(1.0);
  }

  return 0;
}
