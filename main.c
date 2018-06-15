#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "codi.hpp"


#include "lib.h"
#include "bfgs.h"
#include "braid.h"
#include "braid_test.h"

/* Define the app structure */
typedef struct _braid_App_struct
{
    int      myid;          /* Processor rank*/
    double  *labels;        /* Data: Label vectors (C) */
    double  *theta;         /* theta variables */
    double  *class_weights; /* Weights of the classification problem (W) */
    double  *class_bias;    /* Bias of the classification problem (mu) */
    double  *descentdir;    /* Descent direction (hessian times gradient) */
    double  *gradient;      /* Gradient of objective function wrt theta */
    double  *Hessian;       /* Hessian matrix */
    int     *batch;         /* List of Indicees of the batch elements */
    int      nclasses;      /* Number of classes */
    int      nbatch;        /* Number of elements in the batch */
    int      nchannels;     /* Width of the network */
    int      ntimes;        /* number of time-steps / layers */
    double   gamma;         /* Relaxation parameter   */
    double   deltaT;        /* Time-step size on fine grid */
    double   stepsize;      /* Stepsize for theta updates */
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
    int    ts;
    double tstart, tstop;
    double deltaT;
    
    /* Get the time-step size */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    deltaT = tstop - tstart;
 
    /* Get the current time index */
    braid_StepStatusGetTIndex(status, &ts);
 

    /* Take one step */
    take_step(u->Ytrain, app->theta, ts, deltaT, app->batch, app->nbatch, app->nchannels, 0);

 
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
       dot += pow( getValue(u->Ytrain[i]), 2 );
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
        sprintf(filename, "%s.%02d", "Yout.pint.myid", app->myid);
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
    double *dbuffer   = (double*) buffer;
    int          nchannels = app->nchannels;
    int          nbatch    = app->nbatch;
    
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
    my_Vector   *u         = NULL;
    double *dbuffer   = (double*) buffer;
    int          nchannels = app->nchannels;
    int          nbatch    = app->nbatch;
 
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
    int    nbatch    = app->nbatch;
    int    nchannels = app->nchannels;
    int    ntimes    = app->ntimes;
    int    nclasses  = app->nclasses;
    double tmp;
    double obj = 0.0;
    int    ts;
 
    /* Get the time index*/
    braid_ObjectiveStatusGetTIndex(ostatus, &ts);
 
    if (ts < ntimes)
    {
        /* Compute regularization term */
        tmp = app->gamma * regularization(app->theta, ts, app->deltaT, ntimes, nchannels);
        obj = tmp;
    }
    else
    {
        /* Evaluate loss */
       tmp  = 1./nbatch * loss(u->Ytrain, app->labels, app->batch, nbatch, app->class_weights, app->class_bias, nclasses, nchannels);
       obj = tmp;
    }

    *objective_ptr = getValue(obj);
    
    return 0;
}


int
my_ObjectiveT_diff(braid_App            app,
                  braid_Vector          u,
                  braid_Vector          u_bar,
                  braid_Real            f_bar,
                  braid_ObjectiveStatus ostatus)
{
    int    nbatch    = app->nbatch;
    int    nchannels = app->nchannels;
    int    ntimes    = app->ntimes;
    int    nclasses  = app->nclasses;
    int    ntheta    = (nchannels * nchannels + 1 ) * ntimes;
    int    nstate    = nchannels * nbatch;
    int    ts;
    RealReverse obj;
 
    /* Get the time index*/
    braid_ObjectiveStatusGetTIndex(ostatus, &ts); 

    /* Set up CoDiPack */
    RealReverse::TapeType& codiTape = RealReverse::getGlobalTape();
    codiTape.setActive();

    /* Register input */
    RealReverse* Ycodi;  /* CodiType for the state */
    RealReverse* theta;  /* CodiType for the theta */
    Ycodi  = (RealReverse*) malloc(nstate  * sizeof(RealReverse));
    theta = (RealReverse*) malloc(ntheta * sizeof(RealReverse));
    for (int i = 0; i < nstate; i++)
    {
        Ycodi[i] = u->Ytrain[i];
        codiTape.registerInput(Ycodi[i]);
    }
    /* Register theta as input */
    for (int i = 0; i < ntheta; i++)
    {
        theta[i] = app->theta[i];
        codiTape.registerInput(theta[i]);
    }

    /* Tape the objective function evaluation */
    if (ts < app->ntimes)
    {
        /* Compute regularization term */
        obj = app->gamma * regularization(theta, ts, app->deltaT, ntimes, nchannels);
    }
    else
    {
        /* Evaluate loss at last layer*/
       obj = 1./app->nbatch * loss(Ycodi, app->labels, app->batch,  nbatch, app->class_weights, app->class_bias, nclasses, nchannels);
    } 

    
    /* Set the seed */
    codiTape.setPassive();
    obj.setGradient(f_bar);

    /* Evaluate the tape */
    codiTape.evaluate();

    /* Update adjoint variables and gradient */
    for (int i = 0; i < nstate; i++)
    {
        u_bar->Ytrain[i] = Ycodi[i].getGradient();
    }
    for (int i = 0; i < ntheta; i++)
    {
        app->gradient[i] += theta[i].getGradient();
    }

    /* Reset the codi tape */
    codiTape.reset();

    /* Clean up */
    free(Ycodi);
    free(theta);

   return 0;
}

int
my_Step_diff(braid_App         app,
             braid_Vector      ustop,     /**< input, u vector at *tstop* */
             braid_Vector      u,         /**< input, u vector at *tstart* */
             braid_Vector      ustop_bar, /**< input / output, adjoint vector for ustop */
             braid_Vector      u_bar,     /**< input / output, adjoint vector for u */
             braid_StepStatus  status)
{

    double  tstop, tstart, deltaT;
    int     ts;
    int     nchannels = app->nchannels;
    int     nbatch    = app->nbatch;
    int     ntimes    = app->ntimes;
    int     ntheta   = (nchannels * nchannels + 1 ) * ntimes;
    int     nstate    = nchannels * nbatch;

    /* Get time and time step */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    braid_StepStatusGetTIndex(status, &ts);
    deltaT = tstop - tstart;

    /* Prepare CodiPack Tape */
    RealReverse::TapeType& codiTape = RealReverse::getGlobalTape();
    codiTape.setActive();

    /* Register input */
    RealReverse* Ycodi;  /* CodiType for the state */
    RealReverse* Ynext;  /* CodiType for the state, next time-step */
    RealReverse* theta; /* CodiType for the theta */
    Ycodi  = (RealReverse*) malloc(nstate  * sizeof(RealReverse));
    Ynext  = (RealReverse*) malloc(nstate  * sizeof(RealReverse));
    theta  = (RealReverse*) malloc(ntheta * sizeof(RealReverse));
    for (int i = 0; i < nstate; i++)
    {
        Ycodi[i] = u->Ytrain[i];
        codiTape.registerInput(Ycodi[i]);
        Ynext[i] = Ycodi[i];
    }
    /* Register theta as input */
    for (int i = 0; i < ntheta; i++)
    {
        theta[i] = app->theta[i];
        codiTape.registerInput(theta[i]);
    }
    
    /* Take one forward step */
    take_step(Ynext, theta, ts, deltaT, app->batch, nbatch, nchannels, 0);

    /* Set the adjoint variables */
    codiTape.setPassive();
    for (int i = 0; i < nchannels * nbatch; i++)
    {
        Ynext[i].setGradient(u_bar->Ytrain[i]);
    }

    /* Evaluate the tape */
    codiTape.evaluate();

    /* Update adjoint variables and gradient */
    for (int i = 0; i < nstate; i++)
    {
        u_bar->Ytrain[i] = Ycodi[i].getGradient();
    }
    for (int i = 0; i < ntheta; i++)
    {
        app->gradient[i] += theta[i].getGradient();

    }

    /* Reset the codi tape */
    codiTape.reset();

    /* Clean up */
    free(Ycodi);
    free(Ynext);
    free(theta);

    return 0;
}
 
int 
gradient_access(braid_App app)
{
    int g_idx;
    double nchannels = app->nchannels;
    double ntimes    = app->ntimes;

   /* Print the gradient */
    for (int ts = 0; ts < ntimes; ts++)
    {

        for (int ichannel = 0; ichannel < nchannels; ichannel++)
        {
            
            for (int jchannel = 0; jchannel < nchannels; jchannel++)
            {
                g_idx = ts * (nchannels*nchannels + 1) + ichannel * nchannels + jchannel;
                printf("%d: %02d %03d %1.14e\n", app->myid, ts, g_idx, app->gradient[g_idx]);
            }
        }
        g_idx = ts * (nchannels*nchannels + 1) + nchannels* nchannels;
        printf("%d: %02d %03d %1.14e\n", app->myid, ts, g_idx, app->gradient[g_idx]);
    }

   return 0;
}

int
gradient_allreduce(braid_App app, 
                   MPI_Comm comm)
{

   int ntheta = (app->nchannels * app->nchannels + 1) * app->ntimes;

   double *mygradient = (double*) malloc(ntheta*sizeof(double));
   for (int i = 0; i<ntheta; i++)
   {
       mygradient[i] = app->gradient[i];
   }

   /* Collect sensitivities from all time-processors */
   MPI_Allreduce(mygradient, app->gradient, ntheta, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   free(mygradient);

   return 0;
}                  


int 
my_ResetGradient(braid_App app)
{
    int ntheta = (app->nchannels * app->nchannels + 1) * app->ntimes;

    /* Set the gradient to zero */
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        app->gradient[itheta] = 0.0;
    }

    return 0;
}

int
gradient_norm(braid_App app,
              double   *gradient_norm_prt)

{
    double ntheta = (app->nchannels * app->nchannels + 1) * app->ntimes;
    double gnorm;

    /* Norm of gradient */
    gnorm = 0.0;
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        gnorm += pow(getValue(app->gradient[itheta]), 2);
    }
    gnorm = sqrt(gnorm);

    *gradient_norm_prt = gnorm;
    
    return 0;
}
    
   


int main (int argc, char *argv[])
{
    braid_Core core;
    my_App     *app;

    double   objective;      /**< Objective function */
    double  *labels;         /**< Labels of the data set (C) */
    double  *theta;          /**< theta variables for the network */
    double  *theta0;         /**< Store the old theta variables before linesearch */
    double  *class_weights;  /**< Weights for the classification problem, applied at last layer */
    double  *class_bias;     /**< Bias of the classification problem, applied at last layer */
    double  *gradient;       /**< Gradient of objective function wrt theta */
    double  *gradient0;      /**< Store the old gradient before linesearch */
    double  *descentdir;     /**< Store the old theta variables before linesearch */
    double   gnorm;          /**< Norm of the gradient */
    double   gamma;          /**< Relaxation parameter */
    int     *batch;          /**< Contains indicees of the batch elements */
    int      nclasses;       /**< Number of classes / labels */
    int      nexamples;      /**< Number of elements in the training data */
    int      nbatch;         /**< Size of a batch */
    int      ntheta;        /**< dimension of the theta variables */
    int      ntimes;         /**< Number of layers / time steps */
    int      nchannels;      /**< Number of channels of the netword (width) */
    double   T;              /**< Final time */
    double   theta_init;     /**< Initial theta value */
    double   class_init;     /**< Initial value for the classification weights and biases */
    int      myid;           /**< Processor rank */
    double   deltaT;         /**< Time step size */
    double   stepsize_init;  /**< Initial stepsize for theta updates */
    double  *Hessian;        /**< Hessian matrix */
    double   findiff;        /**< flag: test gradient with finite differences (1) */
    int      maxoptimiter;   /**< Maximum number of optimization iterations */
    double   rnorm;          /**< Space-time Norm of the state variables */
    double   rnorm_adj;      /**< Space-time norm of the adjoint variables */
    double   gtol;           /**< Tolerance for gradient norm */
    double   ls_objective;   /**< Objective function value for linesearch */
    int      ls_maxiter;     /**< Max. number of linesearch iterations */
    double   ls_factor;      /**< Reduction factor for linesearch */
    int      ls_iter;        /**< Iterator for linesearch */
    double  *sk;             /**< BFGS: delta theta */
    double  *yk;             /**< BFGS: delta gradient */
    int      nreq; 
    char     optimfilename[255]; /**< Name of the optimization output file */
    FILE     *optimfile;      /**< File for optimization history */

    /* Problem setup */ 
    nexamples     = 5000;
    nchannels     = 4;
    nclasses      = 5;
    ntimes        = 32;
    deltaT        = 10./32.;     // should be T / ntimes, hard-coded for now due to testing;
    theta_init    = 1e-2;
    class_init    = 1e-1;

    /* Optimization setup */
    gamma         = 1e-2;
    maxoptimiter  = 1;
    gtol          = 1e-4;
    stepsize_init = 1.0;
    ls_maxiter    = 20;
    ls_factor     = 0.5;

    /* Init problem parameters */
    T             = deltaT * ntimes;
    nbatch        = nexamples;
    ntheta        = (nchannels * nchannels + 1 )* ntimes;

    /* Init optimization parameters */
    ls_iter       = 0;
    gnorm         = 0.0;
    rnorm         = 0.0;
    rnorm_adj     = 0.0;

    /* Memory allocation */
    theta         = (double*) malloc(ntheta*sizeof(double));
    theta0        = (double*) malloc(ntheta*sizeof(double));
    class_weights = (double*) malloc(nchannels*nclasses*sizeof(double));
    class_bias    = (double*) malloc(nclasses*sizeof(double));
    descentdir    = (double*) malloc(ntheta*sizeof(double));
    batch         = (int*) malloc(nbatch*sizeof(int));
    gradient      = (double*) malloc(ntheta*sizeof(double));
    gradient0     = (double*) malloc(ntheta*sizeof(double));
    Hessian       = (double*) malloc(ntheta*ntheta*sizeof(double));
    sk            = (double*)malloc(ntheta*sizeof(double));
    yk            = (double*)malloc(ntheta*sizeof(double));


    /* Initialization */
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta[itheta]      = theta_init; 
        theta0[itheta]     = 0.0; 
        descentdir[itheta] = 0.0; 
        gradient[itheta]   = 0.0; 
        gradient0[itheta]  = 0.0; 
    }
    set_identity(ntheta, Hessian);

    /* Initialize the batch (same as examples for now) */
    for (int ibatch = 0; ibatch < nbatch; ibatch++)
    {
        batch[ibatch] = ibatch;
    }

    /* Read the data */
    labels = (double*) malloc(nclasses*nexamples*sizeof(double));
    // read_data("YTarget.transpose.dat", Target, nchannels*nexamples);
    /* Set to zero for now. TODO: Read in the data! */
    for (int i=0; i< nclasses*nexamples; i++)
    {
        labels[i] = 0.0;
    }

    
    /* Initialize classification problem */
    for (int iclasses = 0; iclasses < nclasses; iclasses++)
    {
        for (int ichannels = 0; ichannels < nchannels; ichannels++)
        {
            class_weights[ichannels * nchannels + iclasses] = class_init; 
        }
        class_bias[iclasses] = class_init;
    }

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    /* Set up the app structure */
    app = (my_App *) malloc(sizeof(my_App));
    app->myid          = myid;
    app->labels        = labels;
    app->theta         = theta;
    app->class_weights = class_weights;
    app->class_bias    = class_bias;
    app->gradient      = gradient;
    app->descentdir    = descentdir;
    app->batch         = batch;
    app->nbatch        = nbatch;
    app->nchannels     = nchannels;
    app->nclasses      = nclasses;
    app->ntimes        = ntimes;
    app->deltaT        = deltaT;
    app->gamma         = gamma;
    app->stepsize      = stepsize_init;
    app->Hessian       = Hessian;

    /* Switch for finite difference testing */
    findiff = 1;

    /* Initialize XBraid */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

    /* Initialize adjoint XBraid */
    braid_InitAdjoint( my_ObjectiveT, my_ObjectiveT_diff, my_Step_diff,  my_ResetGradient, &core);

    /* Set some Braid parameters */
    braid_SetMaxLevels(core, 3);
    braid_SetPrintLevel( core, 1);
    braid_SetCFactor(core, -1, 2);
    braid_SetAccessLevel(core, 0);
    braid_SetMaxIter(core, 10);
    braid_SetSkip(core, 0);
    braid_SetAbsTol(core, 1.0e-06);
    braid_SetAbsTolAdjoint(core, 1e-6);


    /* Prepare optimization output */
    if (myid == 0)
    {
       /* Screen output */
    //    printf("\n#    || r ||         || r_adj ||       Objective      || Gradient ||  Stepsize   ls_iter\n");
       
       /* History file */
       sprintf(optimfilename, "%s.%03f.dat", "optim", stepsize_init);
       optimfile = fopen(optimfilename, "w");
       fprintf(optimfile, "#    || r ||         || r_adj ||       Objective      || Gradient ||   Stepsize  ls_iter\n");
    }


    /* Optimization iteration */
    for (int iter = 0; iter < maxoptimiter; iter++)
    {

        /* Parallel-in-time simulation and gradient computation */
        braid_SetObjectiveOnly(core, 0);
        braid_Drive(core);

        /* Get objective function value */
        braid_GetObjective(core, &objective);

        /* Get the state and adjoint residual norms */
        nreq = -1;
        braid_GetRNorms(core, &nreq, &rnorm);
        braid_GetRNormAdjoint(core, &rnorm_adj);

        /* Collect sensitivities from all processors */
        gradient_allreduce(app, MPI_COMM_WORLD);

        /* Compute gradient norm */
        gradient_norm(app, &gnorm);


        // break;

        /* Output */
        if (myid == 0)
        {
            printf("%3d  %1.8e  %1.8e  %1.8e  %8e  %5f  %2d\n", iter, rnorm, rnorm_adj, objective, gnorm, app->stepsize, ls_iter);
            fprintf(optimfile,"%3d  %1.8e  %1.8e  %1.14e  %1.14e  %6f %2d\n", iter, rnorm, rnorm_adj, objective, gnorm, app->stepsize, ls_iter);
            fflush(optimfile);
        }

        /* Check optimization convergence */
        if ( (gnorm < gtol) || (iter == maxoptimiter - 1) )
        {
           break;
        }

        // /* Hessian approximation */
        for (int itheta = 0; itheta < ntheta; itheta++)
        {
            /* Update sk and yk for bfgs */
            sk[itheta] = app->theta[itheta] - theta0[itheta];
            yk[itheta] = app->gradient[itheta] - gradient0[itheta];

            /* Store current theta and gradient vector */
            theta0[itheta]    = app->theta[itheta];
            gradient0[itheta] = app->gradient[itheta];
        }
        bfgs_update(ntheta, sk, yk, app->Hessian);

        /* Compute descent direction */
        double wolfe = 0.0;
        for (int itheta = 0; itheta < ntheta; itheta++)
        {
            /* Compute the descent direction */
            app->descentdir[itheta] = 0.0;
            for (int jtheta = 0; jtheta < ntheta; jtheta++)
            {
                app->descentdir[itheta] -= app->Hessian[itheta*ntheta + jtheta] * app->gradient[jtheta];
            }
            /* compute the wolfe condition product */
            wolfe += app->gradient[itheta] * app->descentdir[itheta];
        }

        /* Backtracking linesearch */
        app->stepsize = stepsize_init;
        for (ls_iter = 0; ls_iter < ls_maxiter; ls_iter++)
        {
            /* Take a trial step using the current stepsize) */
            for (int itheta = 0; itheta < ntheta; itheta++)
            {
                app->theta[itheta] += app->stepsize * app->descentdir[itheta];
            }

            /* Compute new objective function value for that trial step */
            braid_SetObjectiveOnly(core, 1);
            braid_Drive(core);
            braid_GetObjective(core, &ls_objective);

            /* Test the wolfe condition */
            if (ls_objective <= objective + ls_factor * app->stepsize * wolfe ) 
            {
                /* Success, use this theta update */
                break;
            }
            else
            {
                /* Test for line-search failure */
                if (ls_iter == ls_maxiter - 1)
                {
                    printf("\n\n   WARNING: LINESEARCH FAILED! \n\n");
                    break;
                }

                /* Restore the previous theta and gradient variable */
                for (int itheta = 0; itheta < ntheta; itheta++)
                {
                    app->theta[itheta]    = theta0[itheta];
                    app->gradient[itheta] = gradient0[itheta];
                }

                /* Decrease the stepsize */
                app->stepsize = app->stepsize * ls_factor;
            }

        }

        /* Increase stepsize if no reduction has been done */
        if (ls_iter == 0)
        {
            stepsize_init = stepsize_init / ls_factor;
        }

   }


    /* Output */
    if (myid == 0)
    {
        printf("\n Objective     %1.14e", objective);
        printf("\n Gradientnorm: %1.14e", gnorm);
        printf("\n\n");


        /* Print to file */
        write_data("theta_opt.dat", app->theta, ntheta);
        write_data("gradient.dat", app->gradient, ntheta);
    }



    /** ---------------------------------------------------------- 
     * DEBUG: Finite difference testing 
     * ---------------------------------------------------------- */
    if (findiff)
    {
        printf("\n\n------- FINITE DIFFERENCE TESTING --------\n\n");
        double obj_store, obj_perturb;
        double findiff, relerror, max_err;
        double *err = (double*)malloc(ntheta*sizeof(double));
        double EPS    = 1e-3;
        double tolerr = 1e-0;

        /* Store the objective function and the gradient */
        double *grad_store = (double*)malloc(ntheta*sizeof(double));
        for (int idesign = 0; idesign < ntheta; idesign++)
        {
            grad_store[idesign] = app->gradient[idesign];
        }
        braid_GetObjective(core, &obj_store);
        my_ResetGradient(app);

        /* Loop over all design variables */
        for (int idx = 0; idx < ntheta; idx++)
        // int idx = 3;
        {
            /* Perturb the theta */
            app->theta[idx] += EPS;

            /* Run a Braid simulation */
            braid_SetObjectiveOnly(core, 1);
            braid_Drive(core);

            /* Get perturbed objective */
            braid_GetObjective(core, &obj_perturb);

            /* Reset the design */
            app->theta[idx] -= EPS;

            /* Finite differences */
            findiff  = (obj_perturb - obj_store) / EPS;
            relerror = (grad_store[idx] - findiff) / findiff;
            relerror = sqrt(relerror*relerror);
            err[idx] = relerror;
            if (max_err < relerror)
            {
                max_err = relerror;
            }
            printf("\n %d: obj_store %1.14e, obj_perturb %1.14e\n", idx, obj_store, obj_perturb );
            printf("     findiff %1.14e, grad %1.14e, -> ERR %1.14e\n\n", findiff, grad_store[idx], relerror );

            if (fabs(relerror) > tolerr)
            {
                printf("\n\n RELATIVE ERROR TO BIG! DEBUG! \n\n");
                exit(1);
            }

        }
        printf("\n\n MAX. FINITE DIFFERENCES ERROR: %1.14e\n\n", max_err);
        
        free(err);
        free(grad_store);
    }


    /* Clean up */
    free(labels);
    free(Hessian);
    free(theta0);
    free(theta);
    free(class_weights);
    free(class_bias);
    free(gradient0);
    free(gradient);
    free(descentdir);
    free(batch);
    free(sk);
    free(yk);
    free(app);

    braid_Destroy(core);
    MPI_Finalize();

    if (myid == 0)
    {
        fclose(optimfile);
    }


    return 0;
}