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
    double  *Clabels;       /* Data: Label vectors (C) */
    double  *Ydata;         /* Training data */
    double  *theta;         /* theta variables */
    double  *class_W;       /* Weights of the classification problem (W) */
    double  *class_W_grad;  /* Gradient wrt the classification weights */
    double  *class_mu;      /* Bias of the classification problem (mu) */
    double  *class_mu_grad; /* Gradient wrt the classification bias */
    double  *descentdir;    /* Descent direction (hessian times gradient) */
    double  *theta_grad;      /* Gradient of objective function wrt theta */
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
        /* Initialize with training data */
        for (int i = 0; i < nchannels * nbatch; i++)
        {
            u->Ytrain[i] = app->Ydata[i];
        }
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
       tmp  = 1./nbatch * loss(u->Ytrain, app->Clabels, app->batch, nbatch, app->class_W, app->class_mu, nclasses, nchannels);
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
    int nbatch    = app->nbatch;
    int nchannels = app->nchannels;
    int ntimes    = app->ntimes;
    int nclasses  = app->nclasses;
    int ntheta    = (nchannels * nchannels + 1 ) * ntimes;
    int nstate    = nchannels * nbatch;
    int nclassW   = nchannels*nclasses;
    int nclassmu  = nclasses;
    int ts;
    RealReverse obj;
 
    /* Get the time index*/
    braid_ObjectiveStatusGetTIndex(ostatus, &ts); 

    /* Set up CoDiPack */
    RealReverse::TapeType& codiTape = RealReverse::getGlobalTape();
    codiTape.setActive();

    /* Register input */
    RealReverse* Ycodi;   /* CodiType for the state */
    RealReverse* theta;   /* CodiType for the theta */
    RealReverse* class_W; /* CoDiTypye for class_W */
    RealReverse* class_mu; /* CoDiTypye for class_mu */
    Ycodi     = (RealReverse*) malloc(nstate   * sizeof(RealReverse));
    theta     = (RealReverse*) malloc(ntheta   * sizeof(RealReverse));
    class_W   = (RealReverse*) malloc(nclassW  * sizeof(RealReverse));
    class_mu  = (RealReverse*) malloc(nclassmu * sizeof(RealReverse));
    for (int i = 0; i < nstate; i++)
    {
        Ycodi[i] = u->Ytrain[i];
        codiTape.registerInput(Ycodi[i]);
    }
    for (int i = 0; i < ntheta; i++)
    {
        theta[i] = app->theta[i];
        codiTape.registerInput(theta[i]);
    }
    for (int i = 0; i < nclassW; i++)
    {
        class_W[i] = app->class_W[i];
        codiTape.registerInput(class_W[i]);
    }
    for (int i=0; i<nclasses; i++)
    {
        class_mu[i] = app->class_mu[i];
        codiTape.registerInput(class_mu[i]);
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
       obj = 1./app->nbatch * loss(Ycodi, app->Clabels, app->batch,  nbatch, class_W, class_mu, nclasses, nchannels);
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
        app->theta_grad[i] += theta[i].getGradient();
    }
    for (int i = 0; i < nclassW; i++)
    {
        app->class_W_grad[i] += class_W[i].getGradient();
    }
    for (int i=0; i < nclassmu; i++)
    {
        app->class_mu_grad[i] += class_mu[i].getGradient();
    }

    /* Reset the codi tape */
    codiTape.reset();

    /* Clean up */
    free(Ycodi);
    free(theta);
    free(class_W);
    free(class_mu);

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
        app->theta_grad[i] += theta[i].getGradient();

    }

    /* Reset the codi tape */
    codiTape.reset();

    /* Clean up */
    free(Ycodi);
    free(Ynext);
    free(theta);

    return 0;
}
 
// int 
// gradient_access(braid_App app)
// {
//     int g_idx;
//     double nchannels = app->nchannels;
//     double ntimes    = app->ntimes;

//    /* Print the gradient wrt theta */
//     for (int ts = 0; ts < ntimes; ts++)
//     {

//         for (int ichannel = 0; ichannel < nchannels; ichannel++)
//         {
            
//             for (int jchannel = 0; jchannel < nchannels; jchannel++)
//             {
//                 g_idx = ts * (nchannels*nchannels + 1) + ichannel * nchannels + jchannel;
//                 printf("%d: %02d %03d %1.14e\n", app->myid, ts, g_idx, app->theta_grad[g_idx]);
//             }
//         }
//         g_idx = ts * (nchannels*nchannels + 1) + nchannels* nchannels;
//         printf("%d: %02d %03d %1.14e\n", app->myid, ts, g_idx, app->theta_grad[g_idx]);
//     }

//    return 0;
// }

int
gradient_allreduce(braid_App app, 
                   MPI_Comm comm)
{

   int ntheta   = (app->nchannels * app->nchannels + 1) * app->ntimes;
   int nclassW  =  app->nchannels * app->nclasses;
   int nclassmu =  app->nclasses;

   double *theta_grad   = (double*) malloc(ntheta   *sizeof(double));
   double *classW_grad  = (double*) malloc(nclassW  *sizeof(double));
   double *classmu_grad = (double*) malloc(nclassmu *sizeof(double));
   for (int i = 0; i<ntheta; i++)
   {
       theta_grad[i] = app->theta_grad[i];
   }
   for (int i = 0; i < nclassW; i++)
   {
       classW_grad[i] = app->class_W_grad[i];
   }
   for (int i = 0; i < nclassmu; i++)
   {
       classmu_grad[i] = app->class_mu_grad[i];
   }

   /* Collect sensitivities from all time-processors */
   MPI_Allreduce(theta_grad, app->theta_grad, ntheta, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(classW_grad, app->class_W_grad, nclassW, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(classmu_grad, app->class_mu_grad, nclassmu, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   free(theta_grad);
   free(classW_grad);
   free(classmu_grad);

   return 0;
}                  


int 
my_ResetGradient(braid_App app)
{
    int ntheta   = (app->nchannels * app->nchannels + 1) * app->ntimes;
    int nclassW  =  app->nchannels * app->nclasses;
    int nclassmu =  app->nclasses;

    /* Set the gradient to zero */
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        app->theta_grad[itheta] = 0.0;
    }
    for (int i = 0; i < nclassW; i++)
    {
        app->class_W_grad[i] = 0.0;
    }
    for (int i = 0; i < nclassmu; i++)
    {
        app->class_mu_grad[i] = 0.0;
    }


    return 0;
}

int
gradient_norm(braid_App app,
              double   *theta_gnorm_prt,
              double   *class_gnorm_prt)

{
    double ntheta = (app->nchannels * app->nchannels + 1) * app->ntimes;
    int nclassW   =  app->nchannels * app->nclasses;
    int nclassmu  =  app->nclasses;
    double theta_gnorm, class_gnorm;

    /* Norm of gradient */
    theta_gnorm = 0.0;
    class_gnorm = 0.0;
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta_gnorm += pow(getValue(app->theta_grad[itheta]), 2);
    }
    for (int i = 0; i < nclassW; i++)
    {
        class_gnorm += pow(getValue(app->class_W_grad[i]),2);
    }
    for (int i = 0; i < nclassmu; i++)
    {
        class_gnorm += pow(getValue(app->class_mu_grad[i]),2);
    }
    theta_gnorm = sqrt(theta_gnorm);
    class_gnorm = sqrt(class_gnorm);

    *theta_gnorm_prt = theta_gnorm;
    *class_gnorm_prt = class_gnorm;
    
    return 0;
}
    
   


int main (int argc, char *argv[])
{
    braid_Core core;
    my_App     *app;

    double   objective;      /**< Objective function */
    double  *Ydata;          /**< Data set */
    double  *Clabels;        /**< Clabels of the data set (C) */
    double  *theta;          /**< theta variables for the network */
    double  *theta0;         /**< Store the old theta variables before linesearch */
    double  *class_W;        /**< Weights for the classification problem, applied at last layer */
    double  *class_W_grad;   /**< Gradient wrt the classification weights */
    double  *class_mu;       /**< Bias of the classification problem, applied at last layer */
    double  *class_mu_grad;  /**< Gradient wrt the classification bias */
    double  *theta_grad;     /**< Gradient of objective function wrt theta */
    double  *theta_grad0;    /**< Store the old gradient before linesearch */
    double  *descentdir;     /**< Store the old theta variables before linesearch */
    double   theta_gnorm;    /**< Norm of the gradient wrt theta */
    double   class_gnorm;    /**< Norm of the gradient wrt classification weights and bias */
    double   gamma;          /**< Relaxation parameter */
    int     *batch;          /**< Contains indicees of the batch elements */
    int      nclasses;       /**< Number of classes / Clabels */
    int      nexamples;      /**< Number of elements in the training data */
    int      nbatch;         /**< Size of a batch */
    int      ntheta;         /**< dimension of the theta variables */
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
    double   braid_maxlevels;   /**< max. levels of temporal refinement */
    double   braid_printlevel;  /**< print level of xbraid */
    double   braid_cfactor;     /**< temporal coarsening factor */
    double   braid_accesslevel; /**< braid access level */
    double   braid_maxiter;     /**< max. iterations of xbraid */ 
    double   braid_setskip;     /**< braid: skip work on first level */
    double   braid_abstol;      /**< tolerance for primal braid */
    double   braid_abstoladj;   /**< tolerance for adjoint braid */

    int      nreq; 
    char     optimfilename[255]; /**< Name of the optimization output file */
    FILE     *optimfile;      /**< File for optimization history */


    /* --- PROGRAMM SETUP ---*/

    /* Learning problem setup */ 
    nexamples     = 5000;
    nchannels     = 4;
    nclasses      = 5;
    ntimes        = 32;
    deltaT        = 10./32.;     // should be T / ntimes, hard-coded for now due to testing;
    theta_init    = 1e-2;
    class_init    = 1e-1;

    /* Optimization setup */
    gamma         = 1e-2;
    maxoptimiter  = 200;
    gtol          = 1e-4;
    stepsize_init = 1.0;
    ls_maxiter    = 20;
    ls_factor     = 0.5;

    /* XBraid setup */
    braid_maxlevels   = 10;
    braid_printlevel  = 1;
    braid_cfactor     = 2;
    braid_accesslevel = 0;
    braid_maxiter     = 10;
    braid_setskip     = 0;
    braid_abstol      = 1e-10;
    braid_abstoladj   = 1e-6; 

    
    /*--- INITIALIZATION ---*/

    /* Init problem parameters */
    T             = deltaT * ntimes;
    nbatch        = nexamples;
    ntheta        = (nchannels * nchannels + 1 )* ntimes;

    /* Init optimization parameters */
    ls_iter       = 0;
    theta_gnorm   = 0.0;
    class_gnorm   = 0.0;
    rnorm         = 0.0;
    rnorm_adj     = 0.0;

    /* Memory allocation */
    theta         = (double*) malloc(ntheta*sizeof(double));
    theta0        = (double*) malloc(ntheta*sizeof(double));
    class_W       = (double*) malloc(nchannels*nclasses*sizeof(double));
    class_W_grad  = (double*) malloc(nchannels*nclasses*sizeof(double));
    class_mu      = (double*) malloc(nclasses*sizeof(double));
    class_mu_grad = (double*) malloc(nclasses*sizeof(double));
    descentdir    = (double*) malloc(ntheta*sizeof(double));
    batch         = (int*) malloc(nbatch*sizeof(int));
    theta_grad    = (double*) malloc(ntheta*sizeof(double));
    theta_grad0   = (double*) malloc(ntheta*sizeof(double));
    Hessian       = (double*) malloc(ntheta*ntheta*sizeof(double));
    sk            = (double*)malloc(ntheta*sizeof(double));
    yk            = (double*)malloc(ntheta*sizeof(double));
    Clabels       = (double*) malloc(nclasses*nexamples*sizeof(double));
    Ydata         = (double*) malloc(nexamples*nchannels*sizeof(double));

    /* Read the data */
    read_data("Clabels.dat", Clabels, nclasses*nexamples);
    read_data("Ytrain.transpose.dat", Ydata, nchannels*nexamples);


    /* Initialize theta and its gradient */
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta[itheta]        = theta_init; 
        theta0[itheta]       = 0.0; 
        descentdir[itheta]   = 0.0; 
        theta_grad[itheta]   = 0.0; 
        theta_grad0[itheta]  = 0.0; 
    }
    set_identity(ntheta, Hessian);

    /* Initialize the batch (same as examples for now) */
    for (int ibatch = 0; ibatch < nbatch; ibatch++)
    {
        batch[ibatch] = ibatch;
    }


    /* Initialize classification problem */
    for (int iclasses = 0; iclasses < nclasses; iclasses++)
    {
        for (int ichannels = 0; ichannels < nchannels; ichannels++)
        {
            class_W[ichannels * nchannels + iclasses]      = class_init; 
            class_W_grad[ichannels * nchannels + iclasses] = 0.0; 
        }
        class_mu[iclasses]      = class_init;
        class_mu_grad[iclasses] = 0.0;
    }


    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    /* Set up the app structure */
    app = (my_App *) malloc(sizeof(my_App));
    app->myid          = myid;
    app->Clabels       = Clabels;
    app->Ydata         = Ydata;
    app->theta         = theta;
    app->theta_grad    = theta_grad;
    app->class_W       = class_W;
    app->class_W_grad  = class_W_grad;
    app->class_mu      = class_mu;
    app->class_mu_grad = class_mu_grad;
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

    /* Initialize XBraid */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

    /* Initialize adjoint XBraid */
    braid_InitAdjoint( my_ObjectiveT, my_ObjectiveT_diff, my_Step_diff,  my_ResetGradient, &core);

    /* Set Braid parameters */
    braid_SetMaxLevels(core, braid_maxlevels);
    braid_SetPrintLevel( core, braid_printlevel);
    braid_SetCFactor(core, -1, braid_cfactor);
    braid_SetAccessLevel(core, braid_accesslevel);
    braid_SetMaxIter(core, braid_maxiter);
    braid_SetSkip(core, braid_setskip);
    braid_SetAbsTol(core, braid_abstol);
    braid_SetAbsTolAdjoint(core, braid_abstoladj);


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


    /* --- OPTIMIZATION --- */

    for (int iter = 0; iter < maxoptimiter; iter++)
    {

        /* Parallel-in-time simulation and gradient computation */
        braid_ObjectiveOnly(core, 0);
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
        gradient_norm(app, &theta_gnorm, &class_gnorm);


        /* Output */
        if (myid == 0)
        {
            printf("%3d  %1.8e  %1.8e  %1.8e  %8e %8e %5f  %2d\n", iter, rnorm, rnorm_adj, objective, theta_gnorm, class_gnorm, app->stepsize, ls_iter);
            fprintf(optimfile,"%3d  %1.8e  %1.8e  %1.14e  %1.14e %1.14e  %6f %2d\n", iter, rnorm, rnorm_adj, objective, theta_gnorm, class_gnorm, app->stepsize, ls_iter);
            fflush(optimfile);
        }

        /* Check optimization convergence */
        if (  maximum(theta_gnorm, class_gnorm) < gtol || iter == maxoptimiter - 1 )
        {
           break;
        }

        // /* Hessian approximation */
        for (int itheta = 0; itheta < ntheta; itheta++)
        {
            /* Update sk and yk for bfgs */
            sk[itheta] = app->theta[itheta] - theta0[itheta];
            yk[itheta] = app->theta_grad[itheta] - theta_grad0[itheta];

            /* Store current theta and gradient vector */
            theta0[itheta]    = app->theta[itheta];
            theta_grad0[itheta] = app->theta_grad[itheta];
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
                app->descentdir[itheta] -= app->Hessian[itheta*ntheta + jtheta] * app->theta_grad[jtheta];
            }
            /* compute the wolfe condition product */
            wolfe += app->theta_grad[itheta] * app->descentdir[itheta];
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
            braid_ObjectiveOnly(core, 1);
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
                    app->theta_grad[itheta] = theta_grad0[itheta];
                }

                /* Decrease the stepsize */
                app->stepsize = app->stepsize * ls_factor;
            }

        }

        /* Increase stepsize if no reduction has been done */
        // if (ls_iter == 0)
        // {
            // stepsize_init = stepsize_init / ls_factor;
        // }

   }


    /* Output */
    if (myid == 0)
    {
        printf("\n Objective     %1.14e", objective);
        printf("\n Gradientnorm: %1.14e  %1.14e", theta_gnorm, class_gnorm);
        printf("\n\n");


        /* Print to file */
        write_data("theta_opt.dat", app->theta, ntheta);
        write_data("theta_grad.dat", app->theta_grad, ntheta);
        write_data("classW_grad.dat", app->class_W_grad, nchannels * nclasses);
        write_data("classmu_grad.dat", app->class_mu_grad, nclasses);
    }


    /* Switch for finite difference testing */
    findiff = 0;

    /** ---------------------------------------------------------- 
     * DEBUG: Finite difference testing 
     * ---------------------------------------------------------- */
    if (findiff)
    {
        printf("\n\n------- FINITE DIFFERENCE TESTING --------\n\n");
        double obj_store, obj_perturb;
        double findiff, relerror;
        double max_err = 0.0;
        double *err = (double*)malloc(ntheta*sizeof(double));
        double EPS    = 1e-8;
        double tolerr = 1e-0;

        /* Store the objective function and the gradient */
        double *grad_store = (double*)malloc(ntheta*sizeof(double));
        for (int idesign = 0; idesign < ntheta; idesign++)
        {
            grad_store[idesign] = app->theta_grad[idesign];
        }
        braid_GetObjective(core, &obj_store);
        my_ResetGradient(app);

        /* Loop over all design variables */
        // for (int idx = 0; idx < ntheta; idx++)
        int idx = 8;
        {
            /* Perturb the theta */
            app->theta[idx] += EPS;

            /* Run a Braid simulation */
            braid_ObjectiveOnly(core, 1);
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
    free(Clabels);
    free(Ydata);
    free(Hessian);
    free(theta0);
    free(theta);
    free(class_W);
    free(class_W_grad);
    free(class_mu);
    free(class_mu_grad);
    free(theta_grad0);
    free(theta_grad);
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
