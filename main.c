#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "lib.h"
#include "bfgs.h"
#include "braid.h"
#include "braid_wrapper.h"


int main (int argc, char *argv[])
{
    braid_Core core_train;       /**< Braid core for training data */
    braid_Core core_val;         /**< Braid core for validation data */
    my_App     *app;

    double   objective;         /**< Objective function */
    double  *Ytrain;            /**< Traning data set */
    double  *Ctrain;            /**< Classes of the training data set */
    double  *Yval;              /**< Validation data set */
    double  *Cval;              /**< Classes of the validation data set */
    double  *theta;             /**< theta variables for the network */
    double  *theta_grad;        /**< Gradient of objective function wrt theta */
    double  *classW;           /**< Weights for the classification problem, applied at last layer */
    double  *classW_grad;      /**< Gradient wrt the classification weights */
    double  *classMu;          /**< Bias of the classification problem, applied at last layer */
    double  *classMu_grad;     /**< Gradient wrt the classification bias */
    double   gamma_theta;       /**< Relaxation parameter for theta */
    double   gamma_class;       /**< Relaxation parameter for the classification weights and bias */
    int      nclasses;          /**< Number of classes / Clabels */
    int      ntraining;         /**< Number of examples in the training data */
    int      nvalidation;       /**< Number of examples in the validation data */
    int      ntheta;            /**< dimension of the theta variables */
    int      ndesign;           /**< Number of global design variables (theta, classW and classMu) */
    int      ntimes;            /**< Number of layers / time steps */
    int      nchannels;         /**< Number of channels of the netword (width) */
    double   T;                 /**< Final time */
    double   theta_init;        /**< Initial theta value */
    double   class_init;        /**< Initial value for the classification weights and biases */
    int      myid;              /**< Processor rank */
    double   deltaT;            /**< Time step size */
    double   stepsize;          /**< stepsize for theta updates */
    double   stepsize_init;     /**< Initial stepsize for theta updates */
    double  *Hessian;           /**< Hessian matrix */
    double  *global_design;     /**< All design vars: theta, classW and classMu */
    double  *global_design0;    /**< Old design vector of previous iteration  */
    double  *global_gradient;   /**< Gradient of objective wrt all design vars: theta, classW and classMu */
    double  *global_gradient0;  /**< Old gradient at previous iteration */
    double  *descentdir;       /**< Descent direction for optimization algorithm  */
    double   gnorm;             /**< Norm of the global gradient */
    double   findiff;           /**< flag: test gradient with finite differences (1) */
    int      maxoptimiter;      /**< Maximum number of optimization iterations */
    double   rnorm;             /**< Space-time Norm of the state variables */
    double   rnorm_adj;         /**< Space-time norm of the adjoint variables */
    double   gtol;              /**< Tolerance for gradient norm */
    double   ls_objective;      /**< Objective function value for linesearch */
    int      ls_maxiter;        /**< Max. number of linesearch iterations */
    double   ls_factor;         /**< Reduction factor for linesearch */
    int      ls_iter;           /**< Iterator for linesearch */
    double   wolfe;             /**< Wolfe conditoin for linesearch */
    int      braid_maxlevels;   /**< max. levels of temporal refinement */
    int      braid_printlevel;  /**< print level of xbraid */
    int      braid_cfactor;     /**< temporal coarsening factor */
    int      braid_accesslevel; /**< braid access level */
    int      braid_maxiter;     /**< max. iterations of xbraid */ 
    int      braid_setskip;     /**< braid: skip work on first level */
    double   braid_abstol;      /**< tolerance for primal braid */
    double   braid_abstoladj;   /**< tolerance for adjoint braid */
    double   accur_train;       /**< Prediction accuracy on the training data */
    double   accur_val;         /**< Prediction accuracy on the validation data */

    char     optimfilename[255]; /**< Name of the optimization output file */
    FILE     *optimfile;      /**< File for optimization history */
    int      nreq, arg_index; 

    struct rusage r_usage;
    double StartTime, StopTime, UsedTime;


    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    StartTime = MPI_Wtime();

    /* --- PROGRAMM SETUP (Default parameters) ---*/

    /* Learning problem setup */ 
    ntraining     = 5000;
    nvalidation   = 1000;
    nchannels     = 4;
    nclasses      = 5;
    ntimes        = 32;
    deltaT        = 10./32.;     // should be T / ntimes, hard-coded for now due to testing;
    theta_init    = 1e-2;
    class_init    = 1e-1;

    /* Optimization setup */
    gamma_theta   = 1e-2;
    gamma_class   = 1e-5;
    maxoptimiter  = 50;
    gtol          = 1e-4;
    stepsize_init = 1.0;
    ls_maxiter    = 20;
    ls_factor     = 0.5;

    /* XBraid setup */
    braid_maxlevels   = 1;
    braid_printlevel  = 1;
    braid_cfactor     = 2;
    braid_accesslevel = 0;
    braid_maxiter     = 10;
    braid_setskip     = 0;
    braid_abstol      = 1e-10;
    braid_abstoladj   = 1e-6; 



    /* Parse command line */
    arg_index = 1;
    while (arg_index < argc)
    {
        if ( strcmp(argv[arg_index], "-help") == 0 )
        {
           if ( myid == 0 )
           {
              printf("\n");
              printf("USAGE  -nl     <number of layers>             (Default: 32)  \n");
              printf("       -cf     <coarsening factor>            (Default: 2)   \n");
              printf("       -ml     <max. xbraid levels>           (Default: 1)   \n");
              printf("       -mbi    <max. xbraid iterations>       (Default: 10   \n");
              printf("       -btol   <xbraid tolerance>             (Default: 1e-10\n");
              printf("       -abtol  <adjoint xbraid tolerance>     (Default: 1e-6\n");
              printf("       -moi    <max. optimization iterations> (Default: 50)  \n");
              printf("       -mli    <max. linesearch iterations>   (Default: 20)  \n");
              printf("       -pl     <print level>                  (Default is 1) \n");
              printf("\n");
           }
           exit(1);
        }
        else if ( strcmp(argv[arg_index], "-nl") == 0 )
        {
           arg_index++;
           ntimes = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-cf") == 0 )
        {
           arg_index++;
           braid_cfactor = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-ml") == 0 )
        {
           arg_index++;
           braid_maxlevels = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-mbi") == 0 )
        {
           arg_index++;
           braid_maxiter = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-btol") == 0 )
        {
           arg_index++;
           braid_abstol = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-abtol") == 0 )
        {
           arg_index++;
           braid_abstoladj = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-moi") == 0 )
        {
           arg_index++;
           maxoptimiter = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-mli") == 0 )
        {
           arg_index++;
           ls_maxiter = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-pl") == 0 )
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
    ndesign        = ntheta + nchannels * nclasses + nclasses;

    /* Init optimization parameters */
    ls_iter       = 0;
    gnorm         = 0.0;
    objective     = 0.0;
    rnorm         = 0.0;
    rnorm_adj     = 0.0;
    stepsize      = stepsize_init;

    /* Memory allocation */
    theta             = (double*) malloc(ntheta*sizeof(double));
    theta_grad        = (double*) malloc(ntheta*sizeof(double));
    classW           = (double*) malloc(nchannels*nclasses*sizeof(double));
    classW_grad      = (double*) malloc(nchannels*nclasses*sizeof(double));
    classMu          = (double*) malloc(nclasses*sizeof(double));
    classMu_grad     = (double*) malloc(nclasses*sizeof(double));
    Hessian           = (double*) malloc(ndesign*ndesign*sizeof(double));
    global_design     = (double*) malloc(ndesign*sizeof(double));
    global_design0    = (double*) malloc(ndesign*sizeof(double));
    global_gradient   = (double*) malloc(ndesign*sizeof(double));
    global_gradient0  = (double*) malloc(ndesign*sizeof(double));
    descentdir        = (double*) malloc(ndesign*sizeof(double));
    Ctrain            = (double*) malloc(nclasses*ntraining*sizeof(double));
    Ytrain            = (double*) malloc(ntraining*nchannels*sizeof(double));
    Cval              = (double*) malloc(nclasses*nvalidation*sizeof(double));
    Yval              = (double*) malloc(nvalidation*nchannels*sizeof(double));

    /* Read the training and validation data */
    read_data("data/Ytrain.dat", Ytrain, nchannels*ntraining);
    read_data("data/Ctrain.dat", Ctrain, nclasses*ntraining);
    read_data("data/Yval.dat", Yval, nchannels*nvalidation);
    read_data("data/Cval.dat", Cval, nclasses*nvalidation);


    /* Initialize theta and its gradient */
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta[itheta]            = theta_init * itheta; 
        theta_grad[itheta]       = 0.0; 
    }

    /* Initialize classification parameters and gradient */
    for (int iclasses = 0; iclasses < nclasses; iclasses++)
    {
        for (int ichannels = 0; ichannels < nchannels; ichannels++)
        {
            classW[iclasses * nchannels + ichannels]       = class_init * iclasses * ichannels; 
            classW_grad[iclasses * nchannels + ichannels]  = 0.0; 
        }
        classMu[iclasses]       = class_init * iclasses;
        classMu_grad[iclasses]  = 0.0;
    }

    /* Initialize optimization variables */
    for (int idesign = 0; idesign < ndesign; idesign++)
    {
        global_design[idesign]    = 0.0;
        global_design0[idesign]   = 0.0;
        global_gradient[idesign]  = 0.0;
        global_gradient0[idesign] = 0.0;
        descentdir[idesign]       = 0.0; 
    }
    set_identity(ndesign, Hessian);
    concat_3vectors(ntheta, theta, nchannels*nclasses, classW, nclasses, classMu, global_design);

    /* Set up the app structure */
    app = (my_App *) malloc(sizeof(my_App));
    app->myid              = myid;
    app->Ctrain            = Ctrain;
    app->Ytrain            = Ytrain;
    app->Cval              = Cval;
    app->Yval              = Yval;
    app->theta             = theta;
    app->theta_grad        = theta_grad;
    app->classW           = classW;
    app->classW_grad      = classW_grad;
    app->classMu          = classMu;
    app->classMu_grad     = classMu_grad;
    app->ntraining         = ntraining;
    app->nvalidation       = nvalidation;
    app->nclasses          = nclasses;
    app->nchannels         = nchannels;
    app->ntimes            = ntimes;
    app->gamma_theta       = gamma_theta;
    app->gamma_class       = gamma_class;
    app->deltaT            = deltaT;
    app->accuracy          = 0;
    app->output            = 0;

    /* Initialize (adjoint) XBraid for training data set */
    app->training = 1;
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_train);
    braid_InitAdjoint( my_ObjectiveT, my_ObjectiveT_diff, my_Step_diff,  my_ResetGradient, &core_train);

    /* Initialize (adjoint) XBraid for validation data set */
    app->training = 0;
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, ntimes, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_val);
    braid_InitAdjoint( my_ObjectiveT, my_ObjectiveT_diff, my_Step_diff,  my_ResetGradient, &core_val);

    /* Set Braid parameters */
    braid_SetMaxLevels(core_train, braid_maxlevels);
    braid_SetMaxLevels(core_val,   braid_maxlevels);
    braid_SetPrintLevel( core_train, braid_printlevel);
    braid_SetPrintLevel( core_val,   braid_printlevel);
    braid_SetCFactor(core_train, -1, braid_cfactor);
    braid_SetCFactor(core_val,   -1, braid_cfactor);
    braid_SetAccessLevel(core_train, braid_accesslevel);
    braid_SetAccessLevel(core_val,   braid_accesslevel);
    braid_SetMaxIter(core_train, braid_maxiter);
    braid_SetMaxIter(core_val,   braid_maxiter);
    braid_SetSkip(core_train, braid_setskip);
    braid_SetSkip(core_val,   braid_setskip);
    braid_SetAbsTol(core_train, braid_abstol);
    braid_SetAbsTol(core_val,   braid_abstol);
    braid_SetAbsTolAdjoint(core_train, braid_abstoladj);
    braid_SetAbsTolAdjoint(core_val,   braid_abstoladj);


    /* Prepare optimization output */
    if (myid == 0)
    {
       /* Screen output */
       printf("\n#    || r ||       || r_adj ||      Objective     || grad ||     Stepsize   ls_iter   Accuracy_train  Accuracy_val\n");
       
       /* History file */
       sprintf(optimfilename, "%s.dat", "optim");
       optimfile = fopen(optimfilename, "w");
       fprintf(optimfile, "#    || r ||         || r_adj ||     Objective          || grad ||       Stepsize  ls_iter   Accuracy_train  Accuracy_val\n");
    }

    // app->theta[3] += 1e-4;

    /* --- OPTIMIZATION --- */

    for (int iter = 0; iter < maxoptimiter; iter++)
    {

        /* --- Training data: Objective function evaluation and gradient computation ---*/ 

        /* Parallel-in-layer propagation and gradient computation  */
        braid_SetObjectiveOnly(core_train, 0);
        app->training = 1;
        braid_Drive(core_train);

        /* Get objective function value and prediction accuracy for training data */
        braid_GetObjective(core_train, &objective);
        MPI_Allreduce(&app->accuracy, &accur_train, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* Get the state and adjoint residual norms */
        nreq = -1;
        braid_GetRNorms(core_train, &nreq, &rnorm);
        braid_GetRNormAdjoint(core_train, &rnorm_adj);

        /* Collect sensitivity information into the 'global_gradient' vector */
        collect_gradient(app, MPI_COMM_WORLD, global_gradient);

        /* Compute gradient norm */
        gnorm = vector_norm(ndesign, global_gradient);


        /* --- Compute Validation Accuracy --- */

        /* Propagate validation data */
        braid_SetObjectiveOnly(core_val, 1);
        app->training = 0;
        braid_Drive(core_val);

        /* Get prediction accuracy for validation data */
        MPI_Allreduce(&app->accuracy, &accur_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        /* --- Optimization control and output ---*/

        /* Check optimization convergence */
        if (  gnorm < gtol )
        {
           break;
        }

        /* Output */
        if (myid == 0)
        {
            printf("%3d  %1.8e  %1.8e  %1.8e  %8e  %5f  %2d   %2.1f%%  %2.1f%%\n", iter, rnorm, rnorm_adj, objective, gnorm, stepsize, ls_iter, accur_train, accur_val);
            fprintf(optimfile,"%3d  %1.8e  %1.8e  %1.14e  %1.14e  %6f  %2d  %2.1f%%  %2.1f%%\n", iter, rnorm, rnorm_adj, objective, gnorm, stepsize, ls_iter, accur_train, accur_val);
            fflush(optimfile);
        }


        /* --- Design update --- */

        /* Hessian approximation */
        bfgs(ndesign, global_design, global_design0, global_gradient, global_gradient0, Hessian);

        /* Compute descent direction for the design and wolfe condition */
        wolfe = compute_descentdir(ndesign, Hessian, global_gradient, descentdir);

        /* Store current design and gradient into *0 vectors */
        copy_vector(ndesign, global_design, global_design0);
        copy_vector(ndesign, global_gradient, global_gradient0);

        /* Backtracking linesearch */
        stepsize = stepsize_init;
        for (ls_iter = 0; ls_iter < ls_maxiter; ls_iter++)
        {

            /* Take a trial step using the current stepsize) */
            update_design(ndesign, stepsize, descentdir, global_design);
            split_into_3vectors(global_design, ntheta, app->theta, nchannels*nclasses, app->classW, nclasses, app->classMu);

            /* Compute new objective function value for that trial step */
            braid_SetPrintLevel(core_train, 0);
            braid_SetObjectiveOnly(core_train, 1);
            app->training = 1;
            braid_Drive(core_train);
            braid_GetObjective(core_train, &ls_objective);
            braid_SetPrintLevel( core_train, braid_printlevel);

            if (myid == 0) printf("ls_iter %d, ls_objective %1.14e\n", ls_iter, ls_objective);

            /* Test the wolfe condition */
            if (ls_objective <= objective + ls_factor * stepsize * wolfe ) 
            {
                /* Success, use this new theta -> keep it in app->theta */
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

                /* Restore the previous design */
                copy_vector(ndesign, global_design0, global_design);
                split_into_3vectors(global_design, ntheta, app->theta, nchannels*nclasses, app->classW, nclasses, app->classMu);

                /* Decrease the stepsize */
                stepsize = stepsize * ls_factor;
            }

        }

   }


    /* --- Run a final propagation ---- */

    /* Parallel-in-layer propagation and gradient computation  */
    braid_SetObjectiveOnly(core_train, 1);
    app->training = 1;
    braid_Drive(core_train);

    /* Get objective function value and prediction accuracy for training data */
    braid_GetObjective(core_train, &objective);
    MPI_Allreduce(&app->accuracy, &accur_train, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* --- Output --- */
    if (myid == 0)
    {
        printf("\n Objective     %1.14e", objective);
        printf("\n Gradientnorm: %1.14e", gnorm);
        printf("\n\n");


        /* Print to file */
        write_data("design_opt.dat", global_design, ndesign);
        // write_data("gradient.dat", global_gradient, ndesign);
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
        braid_GetObjective(core_train, &obj_store);
        my_ResetGradient(app);

        /* Loop over all design variables */
        // for (int idx = 0; idx < ntheta; idx++)
        int idx = 8;
        {
            /* Perturb the theta */
            app->theta[idx] += EPS;

            /* Run a Braid simulation */
            braid_SetObjectiveOnly(core_train, 1);
            app->training = 1;
            braid_Drive(core_train);

            /* Get perturbed objective */
            braid_GetObjective(core_train, &obj_perturb);

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

    StopTime = MPI_Wtime();
    UsedTime = StopTime-StartTime;
    getrusage(RUSAGE_SELF,&r_usage);

    if (myid == 0) 
    {
        printf("Used Time:    %.2f seconds\n", UsedTime);
        printf("Memory Usage: %.2f MB\n",(double) r_usage.ru_maxrss / 1024.0);
    }


    /* Clean up */
    free(Ctrain);
    free(Ytrain);
    free(Cval);
    free(Yval);
    free(Hessian);
    free(global_design);
    free(global_design0);
    free(global_gradient);
    free(global_gradient0);
    free(descentdir);
    free(theta);
    free(theta_grad);
    free(classW);
    free(classW_grad);
    free(classMu);
    free(classMu_grad);

    app->training = 1;
    braid_Destroy(core_train);
    app->training = 0;
    braid_Destroy(core_val);

    free(app);
    MPI_Finalize();

    if (myid == 0)
    {
        fclose(optimfile);
        printf("Optimfile: %s\n", optimfilename);
    }


    return 0;
}
