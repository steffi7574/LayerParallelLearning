#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "lib.h"
#include "braid.h"
#include "braid_wrapper.h"





int main (int argc, char *argv[])
{
    braid_Core core_val;         /**< Braid core for validation data */
    my_App     *app;

    double  *Yval;              /**< Validation data set */
    double  *Cval;              /**< Classes of the validation data set */
    double  *theta;             /**< theta variables for the network */
    double  *classW;           /**< Weights for the classification problem, applied at last layer */
    double  *classMu;          /**< Bias of the classification problem, applied at last layer */
    double  *design_opt;
    double   gamma_theta;       /**< Relaxation parameter for theta */
    double   gamma_class;       /**< Relaxation parameter for the classification weights and bias */
    int      nclasses;          /**< Number of classes / Clabels */
    int      nvalidation;       /**< Number of examples in the validation data */
    int      ntheta;            /**< dimension of the theta variables */
    int      ntimes;            /**< Number of layers / time steps */
    int      nchannels;         /**< Number of channels of the netword (width) */
    int      ndesign;
    double   T;                 /**< Final time */
    int      myid;              /**< Processor rank */
    double   deltaT;            /**< Time step size */
    double   braid_maxlevels;   /**< max. levels of temporal refinement */
    double   braid_printlevel;  /**< print level of xbraid */
    double   braid_cfactor;     /**< temporal coarsening factor */
    double   braid_accesslevel; /**< braid access level */
    double   braid_maxiter;     /**< max. iterations of xbraid */ 
    double   braid_setskip;     /**< braid: skip work on first level */
    double   braid_abstol;      /**< tolerance for primal braid */
    double   braid_abstoladj;   /**< tolerance for adjoint braid */
    double   accur_val;         /**< Prediction accuracy on the validation data */


    /* --- PROGRAMM SETUP ---*/

    /* Learning problem setup */ 
    nvalidation   = 1000;
    nchannels     = 4;
    nclasses      = 5;
    ntimes        = 32;
    deltaT        = 10./32.;     // should be T / ntimes, hard-coded for now due to testing;

    /* Optimization setup */
    gamma_theta   = 1e-2;
    gamma_class   = 1e-5;

    /* XBraid setup */
    braid_maxlevels   = 1;
    braid_printlevel  = 1;
    braid_cfactor     = 2;
    braid_accesslevel = 0;
    braid_maxiter     = 10;
    braid_setskip     = 0;
    braid_abstol      = 1e-10;
    braid_abstoladj   = 1e-6; 

    
    /*--- INITIALIZATION ---*/

    /* Init problem parameters */
    T              = deltaT * ntimes;
    ntheta         = (nchannels * nchannels + 1 )* ntimes;
    ndesign        = ntheta+nchannels*nclasses+nclasses;

    /* Memory allocation */
    theta             = (double*) malloc(ntheta*sizeof(double));
    classW           = (double*) malloc(nchannels*nclasses*sizeof(double));
    classMu          = (double*) malloc(nclasses*sizeof(double));
    Cval              = (double*) malloc(nclasses*nvalidation*sizeof(double));
    Yval              = (double*) malloc(nvalidation*nchannels*sizeof(double));
    design_opt        = (double*) malloc(ndesign*sizeof(double));

    /* Read the validation data */
    read_data("data/Yval.dat", Yval, nchannels*nvalidation);
    read_data("data/Cval.dat", Cval, nclasses*nvalidation);

    /* Read in the optimized design */
    read_data("data/design_opt.dat", design_opt, ndesign);

    /* Initialize theta from optimized design */
    int idesign = 0;
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta[itheta] = design_opt[idesign]; 
        idesign++;
    }
    for (int i=0; i<nchannels*nclasses; i++)
    {
        classW[i] = design_opt[idesign];
        idesign++;
    }
    for (int i = 0; i<nclasses; i++)
    {
       classMu[i] = design_opt[idesign];
       idesign++;
    }

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    /* Set up the app structure */
    app = (my_App *) malloc(sizeof(my_App));
    app->myid              = myid;
    app->Cval              = Cval;
    app->Yval              = Yval;
    app->theta             = theta;
    app->classW           = classW;
    app->classMu          = classMu;
    app->nchannels         = nchannels;
    app->nclasses          = nclasses;
    app->nvalidation       = nvalidation;
    app->ntimes            = ntimes;
    app->deltaT            = deltaT;
    app->gamma_theta       = gamma_theta;
    app->gamma_class       = gamma_class;

    /* Initialize (adjoint) XBraid for validation data set */
    app->training = 0;
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init_Val, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_val);
    braid_InitAdjoint( my_ObjectiveT_Val, my_ObjectiveT_diff, my_Step_diff,  my_ResetGradient, &core_val);

    /* Set Braid parameters */
    braid_SetMaxLevels(core_val,   braid_maxlevels);
    braid_SetPrintLevel( core_val,   braid_printlevel);
    braid_SetCFactor(core_val,   -1, braid_cfactor);
    braid_SetAccessLevel(core_val,   braid_accesslevel);
    braid_SetMaxIter(core_val,   braid_maxiter);
    braid_SetSkip(core_val,   braid_setskip);
    braid_SetAbsTol(core_val,   braid_abstol);
    braid_SetAbsTolAdjoint(core_val,   braid_abstoladj);


    /* --- Compute Validation Accuracy --- */

    /* Prepare propagation of validation data */
    braid_SetObjectiveOnly(core_val, 1);
    app->training = 0;
    /* Propagate validation data */
    braid_Drive(core_val);
    /* Get prediction accuracy for validation data */
    accur_val = app->accuracy;


    /* Output */
    if (myid == 0)
    {
        printf("\n Validation Accuracy: %2.1f%%", accur_val);
        printf("\n\n");

    }



    /* Clean up */
    free(Cval);
    free(Yval);
    free(theta);
    free(classW);
    free(classMu);
    free(app);

    braid_Destroy(core_val);
    MPI_Finalize();


    return 0;
}
