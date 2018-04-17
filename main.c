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
    double *gradient;    /* Gradient of objective function wrt design */
    int    *batch;       /* List of Indicees of the batch elements */
    int     nbatch;      /* Number of elements in the batch */
    int     nchannels;   /* Width of the network */
    int     ntimes;      /* number of time-steps / layers */
    double  gamma;       /* Relaxation parameter   */
    double  theta0;      /* Initial design value */
    double *Ytarget;     /* Target data */
    double  deltaT;      /* Time-step size on fine grid */
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


int 
my_ObjectiveT(braid_App              app,
              braid_Vector           u,
              braid_ObjectiveStatus  ostatus,
              double                *objective_ptr)
{
    double tmp;
    double obj = 0.0;
    int    idx;
 
    /* Get the time index*/
    braid_ObjectiveStatusGetTIndex(ostatus, &idx);
 
    /* Evaluate the objective function at last layer*/
    if ( idx == app->ntimes)
    {
       /* Evaluate objective */
       obj = 1./app->nbatch * loss(u->Ytrain, app->Ytarget, app->batch,  app->nbatch, app->nchannels);
    }
    else
    {
        /* Add regularization term */
        tmp = app->gamma* regularization(app->design, idx, app->deltaT, app->ntimes, app->nchannels);
        obj =  tmp;
    }

    *objective_ptr = obj;
    
    return 0;
}


int
my_ObjectiveT_diff(braid_App            app,
                  braid_Vector          u,
                  braid_Vector          u_bar,
                  braid_Real            f_bar,
                  braid_ObjectiveStatus ostatus)
{
    int    ts;
    int    idx, idx1;
    int    batch_id;
    int    nbatch    = app->nbatch;
    int    nchannels = app->nchannels;
    int    ntimes    = app->ntimes;
    double ddu, ddesign;
 
    /* Get the time index*/
    braid_ObjectiveStatusGetTIndex(ostatus, &ts); 

    if ( ts == app->ntimes)
    {
        for (int ibatch = 0; ibatch < nbatch; ibatch ++)
        {
            /* Get batch_id */
            batch_id = app->batch[ibatch];

            /* Partial derivative of Loss wrt u times f_bar */
            for (int ichannel = 0; ichannel < nchannels; ichannel++)
            {
                idx = batch_id * nchannels + ichannel;
                ddu = u->Ytrain[idx] - app->Ytarget[idx];
                ddu = 1./nbatch * ddu * f_bar;

                /* Update */
                u_bar->Ytrain[idx] += ddu;
            }
        }
    }
    else
    {
       /* Partial derivative of relaxation wrt design times f_bar*/
        /* K(theta)-part */
        for (int ichannel = 0; ichannel < nchannels; ichannel++)
        {
            for (int jchannel = 0; jchannel < nchannels; jchannel++)
            {
                idx  = ts       * (nchannels * nchannels + 1) + ichannel * nchannels + jchannel;
                idx1 = ( ts+1 ) * (nchannels * nchannels + 1) + ichannel * nchannels + jchannel;

                ddesign = app->design[idx];
                if (ts < ntimes - 1)
                {
                    ddesign += -1./(app->deltaT * app->deltaT) * (app->design[idx1] - app->design[idx]);
                }
                ddesign = app->gamma * ddesign * f_bar;
                /* Update */
                app->gradient[idx] += ddesign;
            }
        }

        /* b(theta)-part */
        idx  =   ts     * ( nchannels * nchannels + 1) + nchannels*nchannels;
        ddesign = app->design[idx];
        if (ts < ntimes - 1)
        {
            idx1 = ( ts+1 ) * ( nchannels * nchannels + 1) + nchannels*nchannels;
            ddesign += -1./(app->deltaT * app->deltaT) * (app->design[idx1] - app->design[idx]);
        }
        ddesign = app->gamma * ddesign * f_bar;
        /* Update */
        app->gradient[idx] += ddesign;
    }
  
 
   return 0;
}

int
my_Step_diff(braid_App              app,
                braid_Vector        u,
                braid_Vector        u_bar,
                braid_StepStatus    status)
{

    double tstop, tstart, deltaT;
    int    ts, batch_id;
    int     th_idx, u_idx, ub_idx, g_idx;
    double  Ky, sum, sum_ub, tmp;
    double *ddu;
    int     nchannels = app->nchannels;
    int     nbatch    = app->nbatch;


    /* Get time and time step that have been */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    braid_StepStatusGetTIndex(status, &ts);
    deltaT = tstop - tstart;

    ddu     = (double*)malloc(nchannels * sizeof(double)); 

    /* iterate over all batch elements */ 
    for (int i = 0; i < nbatch; i++)
    {
        batch_id = app->batch[i];
 
        /* Iterate over all channels */
        for (int ichannel = 0; ichannel < nchannels; ichannel++)
        {
            /* Apply differentiated weights and activation */
            sum = 0.0;
            sum_ub = 0.0;
            for (int jchannel = 0; jchannel < nchannels; jchannel++)
            {
                /* Get Ky inside the activation function */
                Ky = 0.0;
                for (int kchannel = 0; kchannel < nchannels; kchannel++)
                {
                    th_idx = ts * (nchannels * nchannels + 1) + kchannel * nchannels + jchannel; 
                    u_idx  = batch_id * nchannels + kchannel;
                    Ky += app->design[th_idx] * u->Ytrain[u_idx];
                }
                tmp = sigma_diff(Ky);

                /* apply K and ub */
                th_idx  = ts * (nchannels * nchannels + 1) + ichannel * nchannels + jchannel; 
                ub_idx  = batch_id * nchannels + jchannel;
                sum    += app->design[th_idx] * tmp * u_bar->Ytrain[ub_idx];
                sum_ub += u_bar->Ytrain[ub_idx];

                /* Gradient update K-part */
                u_idx  = batch_id * nchannels + ichannel;
                ub_idx = batch_id * nchannels + jchannel;
                g_idx  = ts * (nchannels*nchannels + 1) + ichannel * nchannels + jchannel;

                app->gradient[g_idx] += deltaT * tmp * u->Ytrain[u_idx] * u_bar->Ytrain[ub_idx];
            }

            /* Compute the u-update */
            u_idx = batch_id * nchannels + ichannel;
            ddu[ichannel] = u_bar->Ytrain[u_idx] + deltaT * sum;
        }

        /* Gradient update b-part */
        g_idx = ts * (nchannels*nchannels + 1) + nchannels* nchannels;
        app->gradient[g_idx] += deltaT * sum_ub;

        /* Update u */
        for (int ichannel = 0; ichannel < nchannels; ichannel++)
        {
            ub_idx = batch_id * nchannels + ichannel;
            u_bar->Ytrain[ub_idx] = ddu[ichannel];
        }
    }       
 
    free(ddu);

    return 0;
}
 
int 
my_AccessGradient(braid_App app)
{
    int g_idx;
    double nchannels = app->nchannels;
    double ntimes    = app->ntimes;

   /* Print the gradient */
    printf("Gradient:\n"); 

    for (int ts = 0; ts < ntimes; ts++)
    {

        for (int ichannel = 0; ichannel < nchannels; ichannel++)
        {
            
            for (int jchannel = 0; jchannel < nchannels; jchannel++)
            {
                g_idx = ts * (nchannels*nchannels + 1) + ichannel * nchannels + jchannel;
                printf("%d %d %d %d %1.14e\n", ts, ichannel, jchannel, g_idx, app->gradient[g_idx]);
            }
        }
        g_idx = ts * (nchannels*nchannels + 1) + nchannels* nchannels;
        printf("%d     %d %1.14e\n", ts, g_idx, app->gradient[g_idx]);

    }

   return 0;
}

int
my_AllreduceGradient(braid_App app, 
                     MPI_Comm comm)
{

   /* Collect sensitivities from all time-processors */

   return 0;
}                  


int 
my_ResetGradient(braid_App app)
{
    double ndesign = (app->nchannels * app->nchannels + 1) * app->ntimes;

    /* Set the gradient to zero */
    for (int idesign = 0; idesign < ndesign; idesign++)
    {
        app->gradient[idesign] = 0.0;
    }

    return 0;
}

int
my_GradientNorm(braid_App app,
                double   *gradient_norm_prt)

{
    double ndesign = (app->nchannels * app->nchannels + 1) * app->ntimes;
    double gnorm;

    /* Norm of gradient */
    gnorm = 0.0;
    for (int idesign = 0; idesign < ndesign; idesign++)
    {
        gnorm += app->gradient[idesign] * app->gradient[idesign];
    }
    gnorm = sqrt(gnorm);

    *gradient_norm_prt = gnorm;
    
    return 0;
}
    

int 
my_DesignUpdate(braid_App app, 
                double    objective,
                double    rnorm,
                double    rnorm_adj)
{

   /* Hessian approximation */

   /* Design update */

   return 0;
}             


int main (int argc, char *argv[])
{
    braid_Core core;
    my_App     *app;

    double *design;       /**< Design variables for the network */
    double *gradient;     /**< Gradient of objective function wrt design */
    double  gamma;        /**< Relaxation parameter */
    double *Ytarget;      /**< Target data */
    int    *batch;        /**< Contains indicees of the batch elements */
    int     nexamples;    /**< Number of elements in the training data */
    int     nbatch;       /**< Size of a batch */
    int     ndesign;      /**< dimension of the design variables */
    int     ntimes;       /**< Number of layers / time steps */
    int     nchannels;    /**< Number of channels of the netword (width) */
    double  T;            /**< Final time */
    double  theta0;       /**< Initial design value */
    int     myid;         /**< Processor rank */
    double  deltaT;        /**< Time step size */

    /* Problem setup */ 
    nexamples = 5000;
    nchannels = 4;
    ntimes    = 32;
    T         = 10.0;
    theta0    = 1e-2;
    gamma     = 1e-2;

    nbatch  = nexamples;
    ndesign = (nchannels * nchannels + 1 )* ntimes;
    deltaT  = T / ntimes;

    /* Read the target data */
    Ytarget = (double*) malloc(nchannels*nexamples*sizeof(double));
    read_data("Ytarget.transpose.dat", Ytarget, nchannels*nexamples);

    /* Initialize the design and gradient */
    design   = (double*) malloc(ndesign*sizeof(double));
    gradient = (double*) malloc(ndesign*sizeof(double));
    for (int idesign = 0; idesign < ndesign; idesign++)
    {
        design[idesign]   = theta0; 
        gradient[idesign] = 0.0; 
    }

    /* DEBUG: Finite differences */
    // design[16] += 1e-8;

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
    app->gradient  = gradient;
    app->batch     = batch;
    app->nbatch    = nbatch;
    app->nchannels = nchannels;
    app->ntimes    = ntimes;
    app->theta0    = theta0;
    app->deltaT    = deltaT;
    app->Ytarget   = Ytarget;
    app->gamma     = gamma;

    /* Initialize XBraid */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

    /* Initialize adjoint XBraid */
    braid_InitOptimization( my_ObjectiveT, my_Step_diff,  my_ObjectiveT_diff, my_AllreduceGradient, my_ResetGradient,  my_AccessGradient, my_GradientNorm, my_DesignUpdate, &core);

    /* Set some Braid parameters */
    braid_SetPrintLevel( core, 0);
    braid_SetMaxLevels(core, 1);
    braid_SetAbsTol(core, 1.0e-06);
    braid_SetCFactor(core, -1, 2);
    braid_SetAccessLevel(core, 1);
    braid_SetMaxIter(core, 1);
    braid_SetSkip(core, 0);

    braid_SetMaxOptimIter(core, 0);

    /* Run a Braid simulation */
    braid_Drive(core);




    /** ---------------------------------------------------------- 
     * DEBUG: Finite difference testing 
     * Perturb design and run another braid simulation
     * ---------------------------------------------------------- */
    printf("\n\n------- FINITE DIFFERENCE TESTING --------\n\n");
    double obj_orig, obj_perturb;
    double findiff, relerror, err_norm;
    double *err = (double*)malloc(ndesign*sizeof(double));
    double EPS    = 1e-8;
    double tolerr = 1e-2;
    // int    idx = 1;

    err_norm = 0.0;
    for (int idx = 1; idx < ndesign; idx++)
    {

        // int idx = 1;

        /* store the original objective */
        obj_orig = _braid_CoreElt(core, optim)->objective;
        
        /* Perturb the design */
        app->design[idx] += EPS;

        /* Destroy the core and Init a new one core */
        braid_Destroy(core);
        braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);
        braid_InitOptimization( my_ObjectiveT, my_Step_diff,  my_ObjectiveT_diff, my_AllreduceGradient, my_ResetGradient,  my_AccessGradient, my_GradientNorm, my_DesignUpdate, &core);

        /* Set parameters */
        braid_SetPrintLevel( core, 0);
        braid_SetMaxLevels(core, 1);
        braid_SetAbsTol(core, 1.0e-06);
        braid_SetCFactor(core, -1, 2);
        braid_SetAccessLevel(core, 1);
        braid_SetMaxIter(core, 1);
        braid_SetSkip(core, 0);
        braid_SetMaxOptimIter(core, 0);
        braid_SetGradientAccessLevel(core, 0);

       
        /* Reset the gradient from previous run */
        my_ResetGradient(app);
        /* Run a Braid simulation */
        braid_Drive(core);

        /* Get perturbed objective */
        obj_perturb = _braid_CoreElt(core, optim)->objective;

        /* Finite differences */
        findiff  = (obj_perturb - obj_orig) / EPS;
        relerror = (app->gradient[idx] - findiff) / findiff;
        err[idx] = relerror;
        err_norm += relerror*relerror;
        printf("\n %d: obj_orig %1.14e, obj_perturb %1.14e\n", idx, obj_orig, obj_perturb );
        printf("     findiff %1.14e, grad %1.14e, -> ERR %1.14e\n\n", findiff, app->gradient[idx], relerror );

        if (abs(relerror) > tolerr)
        {
            printf("\n\n RELATIVE ERROR TO BIG! DEBUG! \n\n");
            exit(1);
        }

    }
    err_norm = sqrt(err_norm)/ndesign;
    printf("\n\n FINITE DIFFERENCES ERRORNORM: %1.14e\n\n", err_norm);


    free(err);

    /* Clean up */
    free(design);
    free(gradient);
    free(batch);
    free(app);

    braid_Destroy(core);
    MPI_Finalize();


    return 0;
}