#include <stdlib.h>
#include <stdio.h>

#include "defs.hpp"
#include "braid.h"
#include "_braid.h"
#include "network.hpp"
#include "layer.hpp"
#include "dataset.hpp"
#pragma once

/* Define the app structure */
typedef struct _braid_App_struct
{
    int      myid;       /* Processor rank*/
    Network* network;    /* Pointer to the DNN Network Block (local layer storage) */
    DataSet* data;       /* Pointer to the Data set */
    int      ndesign_layermax;  /* Max. number of design vars over all layers */

    braid_Core primalcore; /* Pointer to primal xbraid core, needed for adjoint solve */
} my_App;


/* Define the state vector at one time-step */
typedef struct _braid_Vector_struct
{
   MyReal **state;   /* Network state at one layer, dimensions: nexamples * nchannels */

   Layer* layer;     /* Pointer to layer information (local design part) */

   /* Flag that determines if the layer and state have just been received and thus should be free'd after usage (flag > 0) */
   MyReal sendflag;  
} my_Vector;

/* Set braid options */
void braid_SetConfigOptions(braid_Core core, 
                            Config     *config);


/* Compute time step index from given time */
int GetTimeStepIndex(braid_App app, 
                     MyReal    t);

int GetPrimalIndex(braid_App app,
                   int       ts);

int 
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status);

int
my_Init(braid_App     app,
        MyReal        t,
        braid_Vector *u_ptr);

int
my_Init_diff(braid_App     app,
             MyReal        t,
             braid_Vector  ubar);

int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr);


int
my_Free(braid_App    app,
        braid_Vector u);


int
my_Sum(braid_App     app,
       MyReal        alpha,
       braid_Vector  x,
       MyReal        beta,
       braid_Vector  y);


int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               MyReal       *norm_ptr);


int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus);


int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus);


int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus);


int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus);




int 
my_Step_Adj(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     fstop,
            braid_Vector     u,
            braid_StepStatus status);

int
my_Init_Adj(braid_App     app,
            MyReal        t,
            braid_Vector *u_ptr);

int
my_BufSize_Adj(braid_App           app,
               int                 *size_ptr,
               braid_BufferStatus  bstatus);


int
my_BufPack_Adj(braid_App           app,
               braid_Vector        u,
               void               *buffer,
               braid_BufferStatus  bstatus);


int
my_BufUnpack_Adj(braid_App           app,
                 void               *buffer,
                 braid_Vector       *u_ptr,
                 braid_BufferStatus  bstatus);


void
braid_evalInit(braid_Core core,
         braid_App   app);


void  
braid_evalObjective(braid_Core  core,
                    braid_App   app,     
                    MyReal     *objective,
                    MyReal     *loss_ptr,
                    MyReal     *accuracy_ptr);


void
braid_evalObjectiveDiff(braid_Core core_adj,
                        braid_App  app);


void 
braid_evalInitDiff(braid_Core core_adj,
                   braid_App  app);
