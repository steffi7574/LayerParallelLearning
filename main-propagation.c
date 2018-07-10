#include <sys/resource.h>
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

    double  *Yval;              /**< Data set */
    double  *Cval;              /**< Classes of the Data set */
    double  *theta;             /**< theta variables for the network */
    double  *classW;           /**< Weights for the classification problem, applied at last layer */
    double  *classMu;          /**< Bias of the classification problem, applied at last layer */
    double  *design_opt;
    double   gamma_theta;       /**< Relaxation parameter for theta */
    double   gamma_class;       /**< Relaxation parameter for the classification weights and bias */
    int      nclasses;          /**< Number of classes / Clabels */
    int      nelem;             /**< Number of examples in the data set */
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
    double   accur;             /**< Prediction accuracy on the data */
    double   objective;         /**< Optimization objective function (Loss + Regularization) */
    char   *designdatafilename;
    char   *Ydatafilename;
    char   *Cdatafilename;
    int     arg_index;

    struct rusage r_usage;
    double StartTime, StopTime, UsedTime;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* --- PROGRAMM SETUP ---*/

    /* Learning problem setup */ 
    nelem         = 1000;
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


    /* Default: Propagate validation data */
    designdatafilename = "./optim_STEP3_BFGSall/design_opt.dat";
    Ydatafilename      = "./data/Yval.dat";
    Cdatafilename      = "./data/Cval.dat";


    /* Parse command line */
    arg_index = 1;
    while (arg_index < argc)
    {
        if ( strcmp(argv[arg_index], "-help") == 0 )
        {
           if ( myid == 0 )
           {
              printf("\n");
              printf("USAGE  -nf     <NumberOfFeatureVectors>   \n");
              printf("       -design </path/to/designvariable.dat>   \n");
              printf("       -Ydata  </path/to/featurevectors/Y.dat> \n");
              printf("       -Cdata  </path/to/labelvectors/C.dat>   \n");
              printf("       -nl     <Number of Layers (Default is 32)>   \n");
              printf("       -c      <coarsening factor (Default is 2)>   \n");
              printf("       -ml     <maximum number of levels (Default is 1, serial run)>   \n");
              printf("       -p      <xbraid print level (Default is 1)>   \n");
           }
           exit(1);
        }
        else if ( strcmp(argv[arg_index], "-nf") == 0 )
        {
           arg_index++;
           nelem       = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-design") == 0 )
        {
           arg_index++;
           designdatafilename = argv[arg_index++];
        }else if ( strcmp(argv[arg_index], "-Ydata") == 0 )
        {
           arg_index++;
           Ydatafilename = argv[arg_index++];
        }
        else if ( strcmp(argv[arg_index], "-Cdata") == 0 )
        {
           arg_index++;
           Cdatafilename = argv[arg_index++];
        }
        else if ( strcmp(argv[arg_index], "-nl") == 0 )
        {
           arg_index++;
           ntimes = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-c") == 0 )
        {
           arg_index++;
           braid_cfactor = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-ml") == 0 )
        {
           arg_index++;
           braid_maxlevels = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-p") == 0 )
        {
           arg_index++;
           braid_printlevel = atoi(argv[arg_index++]);
        }
        else
        {
           printf("ABORTING: incorrect command line parameter %s\n", argv[arg_index]);
           MPI_Finalize();
           return (0);
  
        }
    }

    /*--- INITIALIZATION ---*/

    /* Init problem parameters */
    T              = deltaT * ntimes;
    ntheta         = (nchannels * nchannels + 1 )* ntimes;
    ndesign        = ntheta+nchannels*nclasses+nclasses;

    /* Memory allocation */
    theta             = (double*) malloc(ntheta*sizeof(double));
    classW           = (double*) malloc(nchannels*nclasses*sizeof(double));
    classMu          = (double*) malloc(nclasses*sizeof(double));
    Cval              = (double*) malloc(nclasses*nelem*sizeof(double));
    Yval              = (double*) malloc(nelem*nchannels*sizeof(double));
    design_opt        = (double*) malloc(ndesign*sizeof(double));

    /* Read the data that is to be processed */
    read_data(Ydatafilename, Yval, nchannels*nelem);
    read_data(Cdatafilename, Cval, nclasses*nelem);

    /* Read in the optimized design */
    read_data(designdatafilename, design_opt, ndesign);

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
    app->nvalidation       = nelem;
    app->ntimes            = ntimes;
    app->deltaT            = deltaT;
    app->gamma_theta       = gamma_theta;
    app->gamma_class       = gamma_class;
    app->training          = 0;
    app->output            = 1;

    /* Initialize (adjoint) XBraid for validation data set */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_val);
    braid_InitAdjoint( my_ObjectiveT, my_ObjectiveT_diff, my_Step_diff,  my_ResetGradient, &core_val);

    /* Set Braid parameters */
    braid_SetMaxLevels(core_val,   braid_maxlevels);
    braid_SetPrintLevel( core_val,   braid_printlevel);
    braid_SetCFactor(core_val,   -1, braid_cfactor);
    braid_SetAccessLevel(core_val,   braid_accesslevel);
    braid_SetMaxIter(core_val,   braid_maxiter);
    braid_SetSkip(core_val,   braid_setskip);
    braid_SetAbsTol(core_val,   braid_abstol);
    braid_SetAbsTolAdjoint(core_val,   braid_abstoladj);
    braid_SetObjectiveOnly(core_val, 1);

    /* Start Timer */
    StartTime = MPI_Wtime();

    /* --- Compute Prediction Accuracy --- */

    braid_Drive(core_val);
    braid_GetObjective(core_val, &objective);
    MPI_Allreduce(&app->accuracy, &accur, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    /* Output */
    if (myid == 0)
    {
        printf("\n");
        printf(" Objective: %1.14e\n", objective);
        printf(" Accuracy:  %2.1f%%\n", accur);
        printf("\n");

    }



    /* --- Print statistics on this run --- */

    StopTime = MPI_Wtime();
    UsedTime = StopTime-StartTime;
    getrusage(RUSAGE_SELF,&r_usage);
    if (myid == 0) 
    {
        printf("Used Time:    %.2f seconds\n", UsedTime);
        printf("Memory Usage: %.2f MB\n",(double) r_usage.ru_maxrss / 1024.0);
    }



    /* Clean up */
    free(Cval);
    free(Yval);
    free(theta);
    free(classW);
    free(classMu);

    braid_Destroy(core_val);
    free(app);
    
    MPI_Finalize();


    return 0;
}
