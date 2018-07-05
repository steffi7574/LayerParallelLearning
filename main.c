#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "lib.h"
#include "bfgs.h"
#include "braid.h"
#include "braid_wrapper.h"





int main (int argc, char *argv[])
{
    braid_Core core;
    my_App     *app;

    double   objective;         /**< Objective function */
    double  *Ytrain;            /**< Traning data set */
    double  *Ctrain;            /**< Classes of the training data set */
    double  *theta;             /**< theta variables for the network */
    double  *theta0;            /**< Store the old theta variables before linesearch */
    double  *theta_grad;        /**< Gradient of objective function wrt theta */
    double  *theta_grad0;       /**< Store the old gradient before linesearch */
    double  *classW;           /**< Weights for the classification problem, applied at last layer */
    double  *classW0;          /**< Stores old weights before linesearch */
    double  *classW_grad;      /**< Gradient wrt the classification weights */
    double  *classW_grad0;     /**< Stores old gradient before linesearch */
    double  *classMu;          /**< Bias of the classification problem, applied at last layer */
    double  *classMu0;         /**< Stores bias before linesearch */
    double  *classMu_grad;     /**< Gradient wrt the classification bias */
    double  *classMu_grad0;    /**< Store gradient before linesearch */
    double  *descentdir_theta;  /**< Descent direction (hessian times gradient) */
    double   theta_gnorm;       /**< Norm of the gradient wrt theta */
    double   class_gnorm;       /**< Norm of the gradient wrt classification weights and bias */
    double   gamma_theta;       /**< Relaxation parameter for theta */
    double   gamma_class;       /**< Relaxation parameter for the classification weights and bias */
    int     *batch;             /**< Contains indicees of the batch elements */
    int      nclasses;          /**< Number of classes / Clabels */
    int      nexamples;         /**< Number of elements in the training data */
    int      nbatch;            /**< Size of a batch */
    int      ntheta;            /**< dimension of the theta variables */
    int      ntimes;            /**< Number of layers / time steps */
    int      nchannels;         /**< Number of channels of the netword (width) */
    double   T;                 /**< Final time */
    double   theta_init;        /**< Initial theta value */
    double   class_init;        /**< Initial value for the classification weights and biases */
    int      myid;              /**< Processor rank */
    double   deltaT;            /**< Time step size */
    double   stepsize_init;     /**< Initial stepsize for theta updates */
    double  *Hessian;           /**< Hessian matrix */
    double   findiff;           /**< flag: test gradient with finite differences (1) */
    int      maxoptimiter;      /**< Maximum number of optimization iterations */
    double   rnorm;             /**< Space-time Norm of the state variables */
    double   rnorm_adj;         /**< Space-time norm of the adjoint variables */
    double   gtol;              /**< Tolerance for gradient norm */
    double   ls_objective;      /**< Objective function value for linesearch */
    int      ls_maxiter;        /**< Max. number of linesearch iterations */
    double   ls_factor;         /**< Reduction factor for linesearch */
    int      ls_iter;           /**< Iterator for linesearch */
    double  *sk;                /**< BFGS: delta theta */
    double  *yk;                /**< BFGS: delta gradient */
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
    gamma_theta   = 1e-2;
    gamma_class   = 1e-2;
    maxoptimiter  = 100;
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
    theta             = (double*) malloc(ntheta*sizeof(double));
    theta0            = (double*) malloc(ntheta*sizeof(double));
    theta_grad        = (double*) malloc(ntheta*sizeof(double));
    theta_grad0       = (double*) malloc(ntheta*sizeof(double));
    classW           = (double*) malloc(nchannels*nclasses*sizeof(double));
    classW0          = (double*) malloc(nchannels*nclasses*sizeof(double));
    classW_grad      = (double*) malloc(nchannels*nclasses*sizeof(double));
    classW_grad0     = (double*) malloc(nchannels*nclasses*sizeof(double));
    classMu          = (double*) malloc(nclasses*sizeof(double));
    classMu0         = (double*) malloc(nclasses*sizeof(double));
    classMu_grad     = (double*) malloc(nclasses*sizeof(double));
    classMu_grad0    = (double*) malloc(nclasses*sizeof(double));
    descentdir_theta  = (double*) malloc(ntheta*sizeof(double));
    batch             = (int*) malloc(nbatch*sizeof(int));
    Hessian           = (double*) malloc(ntheta*ntheta*sizeof(double));
    sk                = (double*)malloc(ntheta*sizeof(double));
    yk                = (double*)malloc(ntheta*sizeof(double));
    Ctrain            = (double*) malloc(nclasses*nexamples*sizeof(double));
    Ytrain            = (double*) malloc(nexamples*nchannels*sizeof(double));

    /* Read the training data */
    read_data("trainingdata/Ytrain.dat", Ytrain, nchannels*nexamples);
    read_data("trainingdata/Ctrain.dat", Ctrain, nclasses*nexamples);


    /* Initialize theta and its gradient */
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta[itheta]            = theta_init; 
        theta0[itheta]           = 0.0; 
        descentdir_theta[itheta] = 0.0; 
        theta_grad[itheta]       = 0.0; 
        theta_grad0[itheta]      = 0.0; 
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
            classW[ichannels * nchannels + iclasses]       = class_init; 
            classW0[ichannels * nchannels + iclasses]      = 0.0; 
            classW_grad[ichannels * nchannels + iclasses]  = 0.0; 
            classW_grad0[ichannels * nchannels + iclasses] = 0.0; 
        }
        classMu[iclasses]       = class_init;
        classMu0[iclasses]      = 0.0;
        classMu_grad[iclasses]  = 0.0;
        classMu_grad0[iclasses] = 0.0;
    }


    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    /* Set up the app structure */
    app = (my_App *) malloc(sizeof(my_App));
    app->myid              = myid;
    app->Ctrain            = Ctrain;
    app->Ytrain            = Ytrain;
    app->theta             = theta;
    app->theta_grad        = theta_grad;
    app->classW           = classW;
    app->classW_grad      = classW_grad;
    app->classMu          = classMu;
    app->classMu_grad     = classMu_grad;
    app->descentdir_theta  = descentdir_theta;
    app->batch             = batch;
    app->nbatch            = nbatch;
    app->nchannels         = nchannels;
    app->nclasses          = nclasses;
    app->ntimes            = ntimes;
    app->deltaT            = deltaT;
    app->gamma_theta       = gamma_theta;
    app->gamma_class       = gamma_class;
    app->stepsize          = stepsize_init;
    app->Hessian           = Hessian;

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
       printf("\n#    || r ||         || r_adj ||       Objective        || theta_grad ||     || class_grad ||      Stepsize   ls_iter\n");
       
       /* History file */
       sprintf(optimfilename, "%s.dat", "optim");
       optimfile = fopen(optimfilename, "w");
       fprintf(optimfile, "#    || r ||         || r_adj ||     Objective             || theta_grad ||     || class_grad ||      Stepsize  ls_iter\n");
    }


    /* --- OPTIMIZATION --- */

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
        gradient_norm(app, &theta_gnorm, &class_gnorm);


        /* Output */
        if (myid == 0)
        {
            printf("%3d  %1.8e  %1.8e  %1.8e  %8e %8e %5f  %2d\n", iter, rnorm, rnorm_adj, objective, theta_gnorm, class_gnorm, app->stepsize, ls_iter);
            fprintf(optimfile,"%3d  %1.8e  %1.8e  %1.14e  %1.14e %1.14e  %6f %2d\n", iter, rnorm, rnorm_adj, objective, theta_gnorm, class_gnorm, app->stepsize, ls_iter);
            fflush(optimfile);
        }

        /* Check optimization convergence */
        if (  ( theta_gnorm < gtol && class_gnorm < gtol ) || iter == maxoptimiter - 1 )
        {
           break;
        }

        /* Hessian approximation for theta */
        for (int itheta = 0; itheta < ntheta; itheta++)
        {
            /* Update sk and yk for bfgs */
            sk[itheta] = app->theta[itheta] - theta0[itheta];
            yk[itheta] = app->theta_grad[itheta] - theta_grad0[itheta];
        }
        for (int itheta = 0; itheta < ntheta; itheta++)
        {
            /* Store current design theta, classW, classmu and gradient */
            theta0[itheta]      = app->theta[itheta];
            theta_grad0[itheta] = app->theta_grad[itheta];
        }
        bfgs_update(ntheta, sk, yk, app->Hessian);

        /* Compute descent direction for theta */
        double wolfe = 0.0;
        for (int itheta = 0; itheta < ntheta; itheta++)
        {
            /* Compute the descent direction */
            app->descentdir_theta[itheta] = 0.0;
            for (int jtheta = 0; jtheta < ntheta; jtheta++)
            {
                app->descentdir_theta[itheta] -= app->Hessian[itheta*ntheta + jtheta] * app->theta_grad[jtheta];
            }
            /* compute the wolfe condition product */
            wolfe += app->theta_grad[itheta] * app->descentdir_theta[itheta];
        }

        /* Backtracking linesearch */
        app->stepsize = stepsize_init;
        for (ls_iter = 0; ls_iter < ls_maxiter; ls_iter++)
        {
            /* Take a trial step using the current stepsize) */
            update_theta(app, app->stepsize, app->descentdir_theta);

            /* Compute new objective function value for that trial step */
            braid_SetObjectiveOnly(core, 1);
            braid_Drive(core);
            braid_GetObjective(core, &ls_objective);

            /* Test the wolfe condition */
            if (ls_objective <= objective + ls_factor * app->stepsize * wolfe ) 
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

                /* Restore the previous theta and gradient variable */
                for (int itheta = 0; itheta < ntheta; itheta++)
                {
                    app->theta[itheta]      = theta0[itheta];
                    app->theta_grad[itheta] = theta_grad0[itheta];
                }

                /* Decrease the stepsize */
                app->stepsize = app->stepsize * ls_factor;
            }

        }
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
        write_data("classW_grad.dat", app->classW_grad, nchannels * nclasses);
        write_data("classmu_grad.dat", app->classMu_grad, nclasses);
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
    free(Ctrain);
    free(Ytrain);
    free(Hessian);
    free(theta);
    free(theta0);
    free(theta_grad);
    free(theta_grad0);
    free(classW);
    free(classW0);
    free(classW_grad);
    free(classW_grad0);
    free(classMu);
    free(classMu0);
    free(classMu_grad);
    free(classMu_grad0);
    free(descentdir_theta);
    free(batch);
    free(sk);
    free(yk);
    free(app);

    braid_Destroy(core);
    MPI_Finalize();

    if (myid == 0)
    {
        fclose(optimfile);
        printf("Optimfile: %s\n", optimfilename);
    }


    return 0;
}
