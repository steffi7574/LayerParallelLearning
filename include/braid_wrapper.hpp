#include <stdio.h>
#include <stdlib.h>

#include "braid.hpp"
#include "defs.hpp"
// #include "_braid.h"
#include "dataset.hpp"
#include "layer.hpp"
#include "network.hpp"
#pragma once

/**
 * Define the state vector at one time-step
 */
class myBraidVector {
 protected:
  int nbatch;    /* Number of examples */
  int nchannels; /* Number of channels */

  MyReal *
      *state;   /* Network state at one layer, dimensions: nbatch * nchannels */
  Layer *layer; /* Pointer to layer information */

  /* Flag that determines if the layer and state have just been received and
   * thus should be free'd after usage (flag > 0) */
  MyReal sendflag;

 public:
  /* Get dimensions */
  int getnBatch();
  int getnChannels();

  /* Get Pointer to the state at example exampleID */
  MyReal *getState(int exampleID);

  /* Get pointer to the full state matrix */
  MyReal **getState();

  /* Get and set pointer to the layer */
  Layer *getLayer();
  void setLayer(Layer *layer);

  /* Get and set the sendflag */
  MyReal getSendflag();
  void setSendflag(MyReal value);

  /* Constructor */
  myBraidVector(int nChannels, int nBatch);
  /* Destructor */
  ~myBraidVector();
};

/**
 * Wrapper for the primal braid app.
 * virtual function are overwritten from the adjoint app class
 */
class myBraidApp : public BraidApp {
 protected:
  // BraidApp defines tstart, tstop, ntime and comm_t
  int myid;         /* Processor rank*/
  Network *network; /* Pointer to the DNN Network Block (local layer storage) */
  DataSet *data;    /* Pointer to the Data set */

  BraidCore *core; /* Braid core for running PinT simulation */

  /* Output */
  MyReal objective; /* Objective function */

 public:
  /* Constructor */
  myBraidApp(DataSet *Data, Network *Network, Config *Config, MPI_Comm Comm, int current_nlayers);

  /* Destructor */
  ~myBraidApp();

  /* Return objective function */
  MyReal getObjective();

  /* Return the core */
  BraidCore *getCore();

  /* Get xbraid's grid distribution */
  void GetGridDistribution(int *ilower_ptr, int *iupper_ptr);

  /* Return the time step index of current time t */
  braid_Int GetTimeStepIndex(MyReal t);

  /* Apply one time step */
  virtual braid_Int Step(braid_Vector u_, braid_Vector ustop_,
                         braid_Vector fstop_, BraidStepStatus &pstatus);

  /* Compute residual: Does nothing. */
  braid_Int Residual(braid_Vector u_, braid_Vector r_,
                     BraidStepStatus &pstatus);

  /* Allocate a new vector in *v_ptr, which is a deep copy of u_. */
  braid_Int Clone(braid_Vector u_, braid_Vector *v_ptr);

  /* Allocate a new vector in *u_ptr and initialize it with an
 initial guess appropriate for time t. */
  virtual braid_Int Init(braid_Real t, braid_Vector *u_ptr);

  /* De-allocate the vector @a u_. */
  braid_Int Free(braid_Vector u_);

  /* Perform the operation: y_ = alpha * x_ + beta * @a y_. */
  braid_Int Sum(braid_Real alpha, braid_Vector x_, braid_Real beta,
                braid_Vector y_);

  /* Compute in @a *norm_ptr an appropriate spatial norm of @a u_. */
  braid_Int SpatialNorm(braid_Vector u_, braid_Real *norm_ptr);

  /* @see braid_PtFcnAccess. */
  braid_Int Access(braid_Vector u_, BraidAccessStatus &astatus);

  /* @see braid_PtFcnBufSize. */
  virtual braid_Int BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus);

  /* @see braid_PtFcnBufPack. */
  virtual braid_Int BufPack(braid_Vector u_, void *buffer,
                            BraidBufferStatus &bstatus);

  /* @see braid_PtFcnBufUnpack. */
  virtual braid_Int BufUnpack(void *buffer, braid_Vector *u_ptr,
                              BraidBufferStatus &bstatus);

  /* Set the initial condition */
  virtual braid_Int SetInitialCondition();

  /* evaluate objective function */
  virtual braid_Int EvaluateObjective();

  /* Run Braid drive, return norm */
  MyReal run();
};

/**
 * Adjoint braid App for solving adjoint eqations with xbraid.
 */
class myAdjointBraidApp : public myBraidApp {
 protected:
  BraidCore
      *primalcore; /* pointer to primal core for accessing primal states */

 public:
  myAdjointBraidApp(DataSet *Data, Network *Network, Config *config,
                    BraidCore *Primalcoreptr, MPI_Comm comm, 
                    int current_nlayers);

  ~myAdjointBraidApp();

  /* Get the storage index of primal (reversed) */
  int GetPrimalIndex(int ts);

  /* Apply one time step */
  braid_Int Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_,
                 BraidStepStatus &pstatus);

  /* Allocate a new vector in *u_ptr and initialize it with an
 initial guess appropriate for time t. */
  braid_Int Init(braid_Real t, braid_Vector *u_ptr);

  /* @see braid_PtFcnBufSize. */
  braid_Int BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus);

  /* @see braid_PtFcnBufPack. */
  braid_Int BufPack(braid_Vector u_, void *buffer, BraidBufferStatus &bstatus);

  /* @see braid_PtFcnBufUnpack. */
  braid_Int BufUnpack(void *buffer, braid_Vector *u_ptr,
                      BraidBufferStatus &bstatus);

  /* Set the adjoint initial condition (derivative of primal objective function)
   */
  braid_Int SetInitialCondition();

  /* evaluate objective function (being just the derivative of the opening
   * layer) */
  braid_Int EvaluateObjective();
};
