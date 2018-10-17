#include <stdlib.h>
#include <stdio.h>

#include "braid.h"
#include "_braid.h"
#include "network.hpp"
#include "layer.hpp"
#pragma once

/* Define the app structure */
typedef struct _braid_App_struct
{
    int      myid;       /* Processor rank*/
    Network* network;    /* Pointer to the DNN Network */
    int      nexamples;  /* Number of data examples */
    double** examples;   /* Data examples */
    double** labels;     /* Labels for the data examples */

    double   accuracy;   /* Accuracy of the network */
    double   loss;

    braid_Core primalcore; /* Pointer to primal xbraid core, needed for adjoint solve */
} my_App;


/* Define the state vector at one time-step */
typedef struct _braid_Vector_struct
{
   double **state;   /* Network state at one layer, dimensions: nexamples * nchannels */

   double** primal;   /* If adjoint core: Pointer to the primal state, else: NULL */

   Layer* layer;     /* Pointer to layer information */
   /* Flag that determines if the layer and state have just been received and thus should be free'd after usage (flag > 0) */
   double sendflag;  
} my_Vector;


/* Compute time step index from given time */
int GetTimeStepIndex(braid_App app, 
                     double    t);


int 
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status);

int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr);

int
my_Init_diff(braid_App     app,
             double        t,
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
       double        alpha,
       braid_Vector  x,
       double        beta,
       braid_Vector  y);


int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr);


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
my_ObjectiveT(braid_App              app,
              braid_Vector           u,
              braid_ObjectiveStatus  ostatus,
              double                *objective_ptr);


int
my_ObjectiveT_diff(braid_App            app,
                  braid_Vector          u,
                  braid_Vector          u_bar,
                  braid_Real            f_bar,
                  braid_ObjectiveStatus ostatus);

int
my_Step_diff(braid_App         app,
             braid_Vector      ustop,     /**< input, u vector at *tstop* */
             braid_Vector      u,         /**< input, u vector at *tstart* */
             braid_Vector      ustop_bar, /**< input / output, adjoint vector for ustop */
             braid_Vector      u_bar,     /**< input / output, adjoint vector for u */
             braid_StepStatus  status);

int 
my_ResetGradient(braid_App app);



double
evalObjectiveT(braid_App   app,
              braid_Vector u, 
              int          ilayer,
              double       *loss_ptr,
              double       *accuracy_ptr);



int 
my_Step_Adj(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     fstop,
            braid_Vector     u,
            braid_StepStatus status);

int
my_Init_Adj(braid_App     app,
            double        t,
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

