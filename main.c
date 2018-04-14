#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "lib.h"
#include "braid.h"
#include "braid_test.h"

/* Define the app structure */
typedef struct _braid_App_struct
{
    int     myid;        /* Processor rank*/
    double *design;      /* Design variables */
    int    *batch;       /* List of Indicees of the batch elements */
    int     nbatch;      /* Number of elements in the batch */
    int     nchannels;   /* Width of the network */
    int     ntimes;      /* number of time-steps / layers */
    double  alpha;       /* Relaxation parameter   */
    double  theta0;      /* Initial design value */

} my_App;


/* Define the state vector at one time-step */
typedef struct _braid_Vector_struct
{
   double *Ytrain;            /* Training data */

} my_Vector;


int 
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
    int ts;
    double tstart, tstop;
    double deltaT;
    
    /* Get the time-step size */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    deltaT = tstop - tstart;
 
    /* Get the current design */
    braid_StepStatusGetTIndex(status, &ts);
 

    /* Take one step */
    take_step(u->Ytrain, app->design, ts, deltaT, app->batch, app->nbatch, app->nchannels, 0);

 
    /* no refinement */
    braid_StepStatusSetRFactor(status, 1);
 
 
    return 0;
}   


int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{

    my_Vector *u;
    int nchannels = app->nchannels;
    int nbatch    = app->nbatch;
 
    /* Allocate the vector */
    u = (my_Vector *) malloc(sizeof(my_Vector));
    u->Ytrain = (double*) malloc(nchannels * nbatch *sizeof(double));
 
    /* Initialize the vector */
    if (t == 0.0)
    {
        /* Read training data from file */
        read_data("Ytrain.transpose.dat", u->Ytrain, nchannels * nbatch);
    }
    else
    {
        for (int i = 0; i < nchannels * nbatch; i++)
        {
            u->Ytrain[i] = 0.0;
        }
    }

    *u_ptr = u;

    return 0;
}


int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
   my_Vector *v;
   int nchannels = app->nchannels;
   int nbatch    = app->nbatch;
   
   /* Allocate the vector */
   v = (my_Vector *) malloc(sizeof(my_Vector));
   v->Ytrain = (double*) malloc(nchannels * nbatch *sizeof(double));

   /* Clone the values */
    for (int i = 0; i < nchannels * nbatch; i++)
    {
        v->Ytrain[i] = u->Ytrain[i];
    }

   *v_ptr = v;

   return 0;
}


int
my_Free(braid_App    app,
        braid_Vector u)
{
   free(u->Ytrain);
   free(u);

   return 0;
}

int
my_Sum(braid_App     app,
       double        alpha,
       braid_Vector  x,
       double        beta,
       braid_Vector  y)
{
    int nchannels = app->nchannels;
    int nbatch    = app->nbatch;

    for (int i = 0; i < nchannels * nbatch; i++)
    {
       (y->Ytrain)[i] = alpha*(x->Ytrain)[i] + beta*(y->Ytrain)[i];
    }

   return 0;
}

int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
    int nchannels = app->nchannels;
    int nbatch    = app->nbatch;
    double dot;

    dot = 0.0;
    for (int i = 0; i < nchannels * nbatch; i++)
    {
       dot += (u->Ytrain)[i]*(u->Ytrain)[i];
    }
 
   *norm_ptr = sqrt(dot);

   return 0;
}



int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus)
{
    int   idx;
   char  filename[255];

    braid_AccessStatusGetTIndex(astatus, &idx);

    if (idx == app->ntimes)
    {
        sprintf(filename, "%s.%03d", "Yout.pint", app->myid);
        write_data(filename, u->Ytrain, app->nbatch * app->nchannels);
    }

    return 0;
}


int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus)
{
    int nchannels = app->nchannels;
    int nbatch    = app->nbatch;
    
    *size_ptr = nchannels*nbatch*sizeof(double);
    return 0;
}



int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{
    double *dbuffer = buffer;
    int     nchannels = app->nchannels;
    int     nbatch    = app->nbatch;
    
    for (int i = 0; i < nchannels * nbatch; i++)
    {
       dbuffer[i] = (u->Ytrain)[i];
    }
 
    braid_BufferStatusSetSize( bstatus,  nchannels*nbatch*sizeof(double));
 
   return 0;
}



int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus)
{
    my_Vector *u         = NULL;
    double    *dbuffer   = buffer;
    int        nchannels = app->nchannels;
    int        nbatch    = app->nbatch;
 
     /* Allocate the vector */
     u = (my_Vector *) malloc(sizeof(my_Vector));
     u->Ytrain = (double*) malloc(nchannels * nbatch *sizeof(double));

    /* Unpack the buffer */
    for (int i = 0; i < nchannels * nbatch; i++)
    {
       (u->Ytrain)[i] = dbuffer[i];
    }
 
    *u_ptr = u;
    return 0;
}
 

int main (int argc, char *argv[])
{
    braid_Core core;
    my_App     *app;

    double *design;       /**< Design variables for the network */
    int    *batch;        /**< Contains indicees of the batch elements */
    int     nexamples;    /**< Number of elements in the training data */
    int     nbatch;       /**< Size of a batch */
    int     ndesign;      /**< dimension of the design variables */
    int     ntimes;       /**< Number of layers / time steps */
    int     nchannels;    /**< Number of channels of the netword (width) */
    double  T;            /**< Final time */
    double  theta0;       /**< Initial design value */
    int     myid;         /**< Processor rank */

    /* Problem setup */ 
    nexamples = 5000;
    nchannels = 4;
    ntimes    = 32;
    T         = 10.0;
    theta0    = 1e-2;

    nbatch  = nexamples;
    ndesign = (nchannels * nchannels + 1 )* ntimes;

    /* Initialize the design */
    design  = (double*) malloc(ndesign*sizeof(double));
    for (int idesign = 0; idesign < ndesign; idesign++)
    {
        design[idesign] = theta0; 
    }

    /* Initialize the batch (same as examples for now) */
    batch   = (int*) malloc(nbatch*sizeof(int));
    for (int ibatch = 0; ibatch < nbatch; ibatch++)
    {
        batch[ibatch] = ibatch;
    }


    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    /* Set up the app structure */
    app = (my_App *) malloc(sizeof(my_App));
    app->myid      = myid;
    app->design    = design;
    app->batch     = batch;
    app->nbatch    = nbatch;
    app->nchannels = nchannels;
    app->ntimes    = ntimes;
    app->theta0    = theta0;

    /* Initialize XBraid */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

    /* Set some Braid parameters */
    braid_SetPrintLevel( core, 1);
    braid_SetMaxLevels(core, 2);
    braid_SetAbsTol(core, 1.0e-06);
    braid_SetCFactor(core, -1, 2);
    braid_SetAccessLevel(core, 1);
    braid_SetMaxIter(core, 20);
    braid_SetSkip(core, 0);


    /* Run a Braid simulation */
    braid_Drive(core);

    /* Clean up */
    free(design);
    free(batch);
    free(app);

    braid_Destroy(core);
    MPI_Finalize();


    return 0;
}