#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "lib.hpp"
#include "HessianApprox.hpp"
#include "Layer.hpp"
#include "braid.h"
#include "braid_wrapper.hpp"
#include "parser.h"

#define MASTER_NODE 0
#define USE_BFGS  1
#define USE_LBFGS 2

int main (int argc, char *argv[])
{
    braid_Core core_train;       /**< Braid core for training data */
    braid_Core core_val;         /**< Braid core for validation data */
    my_App     *app;

    HessianApprox  *hessian;   /**< Chossing the hessian approximation */
    double   objective;        /**< Objective function */
    double   obj_loss;         /**< Loss term of the objective function */
    double   theta_regul;      /**< Theta-Regulariation term of the objective function */
    double   class_regul;      /**< Classifier-Regulariation term of the objective function */
    double  *Ytrain;           /**< Traning data set */
    double  *Ctrain;           /**< Classes of the training data set */
    double  *Yval;             /**< Validation data set */
    double  *Cval;             /**< Classes of the validation data set */
    double  *theta;            /**< Weights of the network layers */
    double  *theta_grad;       /**< Gradient of objective function wrt theta */
    double  *theta_open;       /**< Weights and bias of the opening layer */
    double  *theta_open_grad;  /**< Gradient of the weights and bias of the opening layer */
    double  *classW;           /**< Weights for the classification problem, applied at last layer */
    double  *classW_grad;      /**< Gradient wrt the classification weights */
    double  *classMu;          /**< Bias of the classification problem, applied at last layer */
    double  *classMu_grad;     /**< Gradient wrt the classification bias */
    double   theta_init;       /**< Factor to scale the initial theta weights and biases */
    double   theta_open_init;  /**< Factor to scale the initial opening layer weights and biases */
    double   class_init;       /**< Factor to scale the initial classification weights and biases */
    double   gamma_theta_tik;  /**< Relaxation parameter for theta tikhonov */
    double   gamma_theta_ddt;  /**< Relaxation parameter for theta time-derivative */
    double   gamma_class;       /**< Relaxation parameter for the classification weights and bias */
    int      nclasses;          /**< Number of classes / Clabels */
    int      ntraining;         /**< Number of examples in the training data */
    int      nvalidation;       /**< Number of examples in the validation data */
    int      nfeatures;         /**< Number of features in the data set */
    int      ntheta_open;       /**< dimension of the opening layer theta variables */
    int      ntheta;            /**< dimension of the theta variables */
    int      nclassW;           /**< dimension of the classification weights W */
    int      ndesign;           /**< Number of global design variables (theta, classW and classMu) */
    int      nlayers;            /**< Number of layers / time steps */
    int      nchannels;         /**< Number of channels of the netword (width) */
    double   T;                 /**< Final time */
    int      myid;              /**< Processor rank */
    int      size;              /**< Number of processors */
    double   deltaT;            /**< Time step size */
    double   stepsize;          /**< stepsize for theta updates */
    double   stepsize_init;     /**< Initial stepsize for theta updates */
    double  *global_design;     /**< All design vars: theta, classW and classMu */
    double  *global_design0;    /**< Old design vector of previous iteration  */
    double  *global_gradient;   /**< Gradient of objective wrt all design vars: theta, classW and classMu */
    double  *global_gradient0;  /**< Old gradient at previous iteration */
    double  *descentdir;       /**< Descent direction for optimization algorithm  */
    double   gnorm;             /**< Norm of the gradient */
    double   mygnorm;           /**< Temporary, holding local norm of the gradient on each proc */
    // double   findiff;           /**< flag: test gradient with finite differences (1) */
    int      maxoptimiter;      /**< Maximum number of optimization iterations */
    double   rnorm;             /**< Space-time Norm of the state variables */
    double   rnorm_adj;         /**< Space-time norm of the adjoint variables */
    double   gtol;              /**< Tolerance for gradient norm */
    double   ls_objective;      /**< Objective function value for linesearch */
    int      ls_maxiter;        /**< Max. number of linesearch iterations */
    double   ls_factor;         /**< Reduction factor for linesearch */
    double   ls_param;          /**< c-parameter for Armijo line-search test */
    int      ls_iter;           /**< Iterator for linesearch */
    int      hessian_approx;     /**< Hessian approximation (1 = BFGS, 2 = L-BFGS) */
    int      lbfgs_stages;       /**< Number of stages of the L-bfgs method */
    double   wolfe;             /**< Wolfe conditoin for linesearch */
    int      braid_maxlevels;   /**< max. levels of temporal refinement */
    int      braid_printlevel;  /**< print level of xbraid */
    int      braid_cfactor;     /**< temporal coarsening factor */
    int      braid_accesslevel; /**< braid access level */
    int      braid_maxiter;     /**< max. iterations of xbraid */ 
    int      braid_setskip;     /**< braid: skip work on first level */
    int      braid_fmg;         /**< braid: V-cycle or full multigrid */
    int      braid_nrelax;      /**< braid: number of CF relaxation sweeps */
    double   braid_abstol;      /**< tolerance for primal braid */
    double   braid_abstoladj;   /**< tolerance for adjoint braid */
    double   accur_train;       /**< Prediction accuracy on the training data */
    double   accur_val;         /**< Prediction accuracy on the validation data */
    int      ReLu;              /**< Flag to determine whether to use ReLu activation or tanh */
    int      openinglayer;      /**< Flag: apply opening layer (1) or just expand data with zero (0) */
    Layer    *layer;            /**< A general layer of the network */
    Layer    *openlayer;        /**< Opening layer: Maps input data to the network width */
    double  (*activation)(double x);  /**< Pointer to the activation function */
    double  (*dactivation)(double x); /**< Pointer to derivative of activation function */

    int      nreq, idx, igrad; 
    char     Ytrain_file[255];
    char     Ctrain_file[255];
    char     Yval_file[255];
    char     Cval_file[255];
    char     optimfilename[255]; /**< Name of the optimization output file */
    FILE    *optimfile;      /**< File for optimization history */

    struct rusage r_usage;
    double StartTime, StopTime, UsedTime, myMB, globalMB; 

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    StartTime = MPI_Wtime();

    /* Data file names */
    sprintf(Ytrain_file, "data/%s.dat", "Ytrain_orig");
    sprintf(Ctrain_file, "data/%s.dat", "Ctrain_orig");
    sprintf(Yval_file,   "data/%s.dat", "Yval_orig");
    sprintf(Cval_file,   "data/%s.dat", "Cval_orig");
    

    /* --- Set DEFAULT parameters for the config option --- */ 


    ntraining         = 5000;
    nvalidation       = 200;
    nfeatures         = 2;
    nclasses          = 5;
    nchannels         = 8;
    nlayers           = 32;
    T                 = 10.0;
    ReLu              = 1;
    openinglayer      = 1;
    braid_cfactor     = 4;
    braid_maxlevels   = 10;
    braid_maxiter     = 3;
    braid_abstol      = 1e-10;
    braid_abstoladj   = 1e-06;
    braid_printlevel  = 1;
    braid_accesslevel = 0;
    braid_setskip     = 0;
    braid_fmg         = 0;
    braid_nrelax      = 1;
    gamma_theta_tik   = 1e-07;
    gamma_theta_ddt   = 1e-07;
    gamma_class       = 1e-05;
    stepsize_init     = 1.0;
    maxoptimiter      = 500;
    gtol              = 1e-08;
    ls_maxiter        = 20;
    ls_factor         = 0.5;
    theta_open_init   = 0.001;
    theta_init        = 0.0;
    class_init        = 0.001;
    hessian_approx    = USE_LBFGS;
    lbfgs_stages      = 20;


    /* --- Read the config file (overwrite default values) --- */

    /* Get config filename from command line argument */
    if (argc != 2)
    {
       if ( myid == MASTER_NODE )
       {
          printf("\n");
          printf("USAGE: ./main </path/to/configfile> \n");
       }
       MPI_Finalize();
       return (0);
    }
    /* Parse the config file */
    config_option_t co;
    if ((co = read_config_file(argv[1])) == NULL) {
        perror("read_config_file()");
        return -1;
    }
    while(1) {

        if ( strcmp(co->key, "ntraining") == 0 )
        {
            ntraining = atoi(co->value);
        }
        else if ( strcmp(co->key, "nvalidation") == 0 )
        {
            nvalidation = atoi(co->value);
        }
        else if ( strcmp(co->key, "nfeatures") == 0 )
        {
            nfeatures = atoi(co->value);
        }
        else if ( strcmp(co->key, "nchannels") == 0 )
        {
            nchannels = atoi(co->value);
        }
        else if ( strcmp(co->key, "nclasses") == 0 )
        {
            nclasses = atoi(co->value);
        }
        else if ( strcmp(co->key, "nlayers") == 0 )
        {
            nlayers = atoi(co->value);
        }
        else if ( strcmp(co->key, "activation") == 0 )
        {
            if ( strcmp(co->value, "ReLu") == 0 )
            {
                activation  = ReLu_act;
                dactivation = d_ReLu_act;
                ReLu = 1;
            }
            else if (strcmp(co->value, "tanh") == 0 )
            {
                activation  = tanh_act;
                dactivation = d_tanh_act;
                ReLu = 0;
            }
            else
            {
                printf("Invalid activation function!");
                MPI_Finalize();
                return(0);
            }
        }
        else if ( strcmp(co->key, "openinglayer") == 0 )
        {
            if ( strcmp(co->value, "YES") == 0 )
            {
                openinglayer = 1;
            }
            else
            {
                openinglayer = 0;
            }
        }
        else if ( strcmp(co->key, "T") == 0 )
        {
            T = atof(co->value);
        }
        else if ( strcmp(co->key, "braid_cfactor") == 0 )
        {
           braid_cfactor = atoi(co->value);
        }
        else if ( strcmp(co->key, "braid_maxlevels") == 0 )
        {
           braid_maxlevels = atoi(co->value);
        }
        else if ( strcmp(co->key, "braid_maxiter") == 0 )
        {
           braid_maxiter = atoi(co->value);
        }
        else if ( strcmp(co->key, "braid_abstol") == 0 )
        {
           braid_abstol = atof(co->value);
        }
        else if ( strcmp(co->key, "braid_adjtol") == 0 )
        {
           braid_abstoladj = atof(co->value);
        }
        else if ( strcmp(co->key, "braid_printlevel") == 0 )
        {
           braid_printlevel = atoi(co->value);
        }
        else if ( strcmp(co->key, "braid_accesslevel") == 0 )
        {
           braid_accesslevel = atoi(co->value);
        }
        else if ( strcmp(co->key, "braid_setskip") == 0 )
        {
           braid_setskip = atoi(co->value);
        }
        else if ( strcmp(co->key, "braid_fmg") == 0 )
        {
           braid_fmg = atoi(co->value);
        }
        else if ( strcmp(co->key, "braid_nrelax") == 0 )
        {
           braid_nrelax = atoi(co->value);
        }
        else if ( strcmp(co->key, "gamma_theta_tik") == 0 )
        {
            gamma_theta_tik = atof(co->value);
        }
        else if ( strcmp(co->key, "gamma_theta_ddt") == 0 )
        {
            gamma_theta_ddt = atof(co->value);
        }
        else if ( strcmp(co->key, "gamma_class") == 0 )
        {
            gamma_class = atof(co->value);
        }
        else if ( strcmp(co->key, "stepsize") == 0 )
        {
            stepsize_init = atof(co->value);
        }
        else if ( strcmp(co->key, "optim_maxiter") == 0 )
        {
           maxoptimiter = atoi(co->value);
        }
        else if ( strcmp(co->key, "gtol") == 0 )
        {
           gtol = atof(co->value);
        }
        else if ( strcmp(co->key, "ls_maxiter") == 0 )
        {
           ls_maxiter = atoi(co->value);
        }
        else if ( strcmp(co->key, "ls_factor") == 0 )
        {
           ls_factor = atof(co->value);
        }
        else if ( strcmp(co->key, "theta_open_init") == 0 )
        {
           theta_open_init = atof(co->value);
        }
        else if ( strcmp(co->key, "theta_init") == 0 )
        {
           theta_init = atof(co->value);
        }
        else if ( strcmp(co->key, "class_init") == 0 )
        {
           class_init = atof(co->value);
        }
        else if ( strcmp(co->key, "hessian_approx") == 0 )
        {
            if ( strcmp(co->value, "BFGS") == 0 )
            {
                hessian_approx = USE_BFGS;
            }
            else if (strcmp(co->value, "L-BFGS") == 0 )
            {
                hessian_approx = USE_LBFGS;
            }
            else
            {
                printf("Invalid Hessian approximation!");
                MPI_Finalize();
                return(0);
            }
        }
        else if ( strcmp(co->key, "lbfgs_stages") == 0 )
        {
           lbfgs_stages = atoi(co->value);
        }
        if (co->prev != NULL) {
            co = co->prev;
        } else {
            break;
        }
    }


    /*--- INITIALIZATION ---*/

    /* Init problem parameters */
    deltaT         = T /(double)nlayers; 
    ntheta_open    = nfeatures * nchannels + 1;
    ntheta         = (nchannels * nchannels + 1 )* nlayers;
    nclassW        = nchannels * nclasses;
    ndesign        = ntheta_open + ntheta + nclassW + nclasses;

    /* Init optimization parameters */
    ls_iter     = 0;
    ls_param    = 1e-4;
    gnorm       = 0.0;
    mygnorm     = 0.0;
    objective   = 0.0;
    obj_loss    = 0.0;
    theta_regul = 0.0;
    class_regul = 0.0;
    rnorm       = 0.0;
    rnorm_adj   = 0.0;
    stepsize    = stepsize_init;

    /* Initialize Hessian approximation */
    if (myid == MASTER_NODE)
    {
        if (hessian_approx == USE_BFGS )
        {
            hessian = new BFGS(ndesign);
        }
        else if (hessian_approx == USE_LBFGS)
        {
            hessian = new L_BFGS(ndesign, lbfgs_stages);
        }
        else
        {
            printf("Invalid Hessian. \n");
            MPI_Finalize();
            return(0);
        }
    }

    /* Read the training and validation data  */
    if (myid == MASTER_NODE)  // Input data only needed on first processor 
    {
        Ytrain = new double [ntraining   * nfeatures];
        Yval   = new double [nvalidation * nfeatures];
        read_data(Ytrain_file, Ytrain, ntraining   * nfeatures);
        read_data(Yval_file,   Yval,   nvalidation * nfeatures);
    }
    if (myid == size - 1) // Labels only needen on last layer 
    {
        Ctrain = new double [ntraining   * nclasses];
        Cval   = new double [nvalidation * nclasses];
        read_data(Ctrain_file, Ctrain, ntraining   * nclasses);
        read_data(Cval_file,   Cval,   nvalidation * nclasses);
    }

    /* Initialize opening layer parameters and its gradient*/
    theta_open       = new double [ntheta_open];
    theta_open_grad  = new double [ntheta_open];
    if (openinglayer)
    {
        for (int ichannels = 0; ichannels < nchannels; ichannels++)
        {
            for (int ifeatures = 0; ifeatures < nfeatures; ifeatures++)
            {
                idx = ichannels * nfeatures + ifeatures;
                theta_open[idx]      = theta_open_init * (double) rand() / ((double) RAND_MAX);
                theta_open_grad[idx] = 0.0;
            }
        }
        idx = nfeatures * nchannels;
        theta_open[idx]      = theta_open_init * (double) rand() / ((double) RAND_MAX);
        theta_open_grad[idx] = 0.0;
    }
    else
    {
        for (int ichannels = 0; ichannels < nchannels; ichannels++)
        {
            for (int ifeatures = 0; ifeatures < nfeatures; ifeatures++)
            {
                idx = ichannels * nfeatures + ifeatures;
                if (ichannels == ifeatures) theta_open[idx] = 1.0;
                else                        theta_open[idx] = 0.0;
                theta_open_grad[idx] = 0.0;
            }
        }
        idx = nfeatures * nchannels;
        theta_open[idx]      = 0.0;
        theta_open_grad[idx] = 0.0;
    }

    /* Initialize classification parameters and its gradient */
    classW       = new double [nclassW];
    classW_grad  = new double [nclassW];
    classMu      = new double [nclasses];
    classMu_grad = new double [nclasses];
    for (int iclasses = 0; iclasses < nclasses; iclasses++)
    {
        for (int ichannels = 0; ichannels < nchannels; ichannels++)
        {
            idx = iclasses * nchannels + ichannels;
            classW[idx]      = class_init * (double) rand() / ((double) RAND_MAX); 
            classW_grad[idx] = 0.0; 
        }
        classMu[iclasses]       = class_init * (double) rand() / ((double) RAND_MAX);
        classMu_grad[iclasses]  = 0.0;
    }

    /* Initialize theta and its gradient */
    theta      = new double [ntheta];
    theta_grad = new double [ntheta];
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        // theta_open[idx]      = theta_init * (double) rand() / ((double) RAND_MAX);
        theta[itheta]      = theta_init * (double) rand() / ((double) RAND_MAX); 
        theta_grad[itheta] = 0.0; 
    }

    /* Initialize global design and gradient only on first processor. */
    if (myid == MASTER_NODE)
    {
        global_design     = new double [ndesign];
        global_design0    = new double [ndesign];
        global_gradient   = new double [ndesign];
        global_gradient0  = new double [ndesign];
        descentdir        = new double [ndesign];
        for (int idesign = 0; idesign < ndesign; idesign++)
        {
            global_design[idesign]    = 0.0;
            global_design0[idesign]   = 0.0;
            global_gradient[idesign]  = 0.0;
            global_gradient0[idesign] = 0.0;
            descentdir[idesign]       = 0.0; 
        }
        concat_4vectors(ntheta_open, theta_open, ntheta, theta, nclassW, classW, nclasses, classMu, global_design);
    }

    /* Initialize the network layers */
    layer     = new DenseLayer(nchannels, activation, dactivation);

    /* Initialize the opening layer */
    if (openinglayer)
    {
        openlayer = new OpenLayer(nchannels, nfeatures, activation, dactivation);
        openlayer->setWeights(theta_open);
        openlayer->setWeights_bar(theta_open_grad);
        openlayer->setBias(&(theta_open[nfeatures*nchannels]));
        openlayer->setBias_bar(&(theta_open_grad[nfeatures*nchannels]));
    }
    else
    {
        openlayer = new OpenLayer(nchannels, nfeatures, NULL, NULL);
    }
        

    /* Set up the app structure */
    app = (my_App *) malloc(sizeof(my_App));
    app->myid            = myid;
    app->Ytrain          = Ytrain;
    app->Ctrain          = Ctrain;
    app->Yval            = Yval;
    app->Cval            = Cval;
    app->theta           = theta;
    app->theta_grad      = theta_grad;
    app->theta_open      = theta_open;
    app->theta_open_grad = theta_open_grad;
    app->classW          = classW;
    app->classW_grad     = classW_grad;
    app->classMu         = classMu;
    app->classMu_grad    = classMu_grad;
    app->ntraining       = ntraining;
    app->nvalidation     = nvalidation;
    app->nfeatures       = nfeatures;
    app->nclasses        = nclasses;
    app->nchannels       = nchannels;
    app->nlayers         = nlayers;
    app->layer           = layer;
    app->openlayer       = openlayer;
    app->gamma_theta_tik = gamma_theta_tik;
    app->gamma_theta_ddt = gamma_theta_ddt;
    app->gamma_class     = gamma_class;
    app->deltaT          = deltaT;
    app->loss            = 0.0;
    app->class_regul     = 0.0;
    app->theta_regul     = 0.0;
    app->accuracy        = 0.0;
    app->output          = 0;

    /* Initialize (adjoint) XBraid for training data set */
    app->training = 1;
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, nlayers, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_train);
    braid_InitAdjoint( my_ObjectiveT, my_ObjectiveT_diff, my_Step_diff,  my_ResetGradient, &core_train);
    if (openinglayer)
    {
        braid_SetInit_diff(core_train, my_Init_diff);
    }

    /* Initialize (adjoint) XBraid for validation data set */
    app->training = 0;
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, nlayers, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_val);
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
    if (braid_fmg){
        braid_SetFMG(core_train);
        braid_SetFMG(core_val);
    }
    braid_SetNRelax(core_train, -1, braid_nrelax);
    braid_SetNRelax(core_val,   -1, braid_nrelax);
    braid_SetAbsTol(core_train, braid_abstol);
    braid_SetAbsTol(core_val,   braid_abstol);
    braid_SetAbsTolAdjoint(core_train, braid_abstoladj);
    braid_SetAbsTolAdjoint(core_val,   braid_abstoladj);


    /* Open and prepare optimization output file*/
    if (myid == MASTER_NODE)
    {
        sprintf(optimfilename, "%s.dat", "optim");
        optimfile = fopen(optimfilename, "w");
        fprintf(optimfile, "# Problem setup: ntraining       %d \n", ntraining);
        fprintf(optimfile, "#                nvalidation     %d \n", nvalidation);
        fprintf(optimfile, "#                nfeatures       %d \n", nfeatures);
        fprintf(optimfile, "#                nclasses        %d \n", nclasses);
        fprintf(optimfile, "#                nchannels       %d \n", nchannels);
        fprintf(optimfile, "#                nlayers         %d \n", nlayers);
        fprintf(optimfile, "#                T               %f \n", T);
        fprintf(optimfile, "#                ReLu activ.?    %d \n", ReLu);
        fprintf(optimfile, "#                opening layer?  %d \n", openinglayer);
        fprintf(optimfile, "# XBraid setup:  max levels      %d \n", braid_maxlevels);
        fprintf(optimfile, "#                coasening       %d \n", braid_cfactor);
        fprintf(optimfile, "#                max. braid iter %d \n", braid_maxiter);
        fprintf(optimfile, "#                abs. tol        %1.e \n", braid_abstol);
        fprintf(optimfile, "#                abs. toladj     %1.e \n", braid_abstoladj);
        fprintf(optimfile, "#                print level     %d \n", braid_printlevel);
        fprintf(optimfile, "#                access level    %d \n", braid_accesslevel);
        fprintf(optimfile, "#                skip?           %d \n", braid_setskip);
        fprintf(optimfile, "#                fmg?            %d \n", braid_fmg);
        fprintf(optimfile, "#                nrelax          %d \n", braid_nrelax);
        fprintf(optimfile, "# Optimization:  gamma_theta_tik %1.e \n", gamma_theta_tik);
        fprintf(optimfile, "#                gamma_theta_ddt %1.e \n", gamma_theta_ddt);
        fprintf(optimfile, "#                gamma_class     %1.e \n", gamma_class);
        fprintf(optimfile, "#                stepsize        %f \n", stepsize_init);
        fprintf(optimfile, "#                max. optim iter %d \n", maxoptimiter);
        fprintf(optimfile, "#                gtol            %1.e \n", gtol);
        fprintf(optimfile, "#                max. ls iter    %d \n", ls_maxiter);
        fprintf(optimfile, "#                ls factor       %f \n", ls_factor);
        fprintf(optimfile, "#                theta_init      %f \n", theta_init);
        fprintf(optimfile, "#                theta_open_init %f \n", theta_open_init);
        fprintf(optimfile, "#                class_init      %f \n", class_init);
        fprintf(optimfile, "#                hessian_approx  %d \n", hessian_approx);
        fprintf(optimfile, "#                lbfgs_stages    %d \n", lbfgs_stages);
        fprintf(optimfile, "\n");
    }

    /* Prepare optimization output */
    if (myid == MASTER_NODE)
    {
       /* Screen output */
       printf("\n#    || r ||          || r_adj ||      Objective       Loss      theta_R   class_R   || grad ||      Stepsize  ls_iter   Accur_train  Accur_val\n");
       
       fprintf(optimfile, "#    || r ||          || r_adj ||      Objective             Loss        theta_reg   class_reg   || grad ||            Stepsize  ls_iter   Accur_train  Accur_val\n");
    }


    /* --- OPTIMIZATION --- */

    for (int iter = 0; iter < maxoptimiter; iter++)
    {

        /* Reset the app */
        app->loss        = 0.0;
        app->theta_regul = 0.0;
        app->class_regul = 0.0;

        /* --- Training data: Objective function evaluation and gradient computation ---*/ 

        /* Parallel-in-layer propagation and gradient computation  */
        braid_SetObjectiveOnly(core_train, 0);
        braid_SetPrintLevel(core_train, 1);
        app->training = 1;
        braid_Drive(core_train);

        /* Get the state and adjoint residual norms */
        nreq = -1;
        braid_GetRNorms(core_train, &nreq, &rnorm);
        braid_GetRNormAdjoint(core_train, &rnorm_adj);

        /* Get objective function and prediction accuracy for training data */
        braid_GetObjective(core_train, &objective);
        MPI_Allreduce(&app->loss, &obj_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&app->theta_regul, &theta_regul, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&app->class_regul, &class_regul, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&app->accuracy, &accur_train, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        /* On Masternode: Get gradient data from app and put it into global_gradient */
        igrad = 0;
        MPI_Reduce(app->theta_open_grad, &(global_gradient[igrad]), ntheta_open, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
        igrad += ntheta_open;
        MPI_Reduce(app->theta_grad, &(global_gradient[igrad]), ntheta, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
        igrad += ntheta;
        MPI_Reduce(app->classW_grad, &(global_gradient[igrad]), nclassW, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
        igrad += nclassW;
        MPI_Reduce(app->classMu_grad, &(global_gradient[igrad]), nclasses, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);

        /* Compute gradient norm */
        mygnorm = 0.0;
        if (myid == MASTER_NODE) {
            mygnorm = vector_normsq(ndesign, global_gradient);
        } 
        MPI_Allreduce(&mygnorm, &gnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gnorm = sqrt(gnorm);


        /* --- Compute Validation Accuracy --- */

        /* Propagate validation data */
        braid_SetObjectiveOnly(core_val, 1);
        braid_SetPrintLevel( core_val,   0);
        app->training = 0;
        braid_Drive(core_val);

        /* Get prediction accuracy for validation data */
        MPI_Allreduce(&app->accuracy, &accur_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        /* --- Optimization control and output ---*/


        /* Output */
        if (myid == MASTER_NODE)
        {
   
            printf("%3d  %1.8e  %1.8e  %1.16e  %1.2e  %1.2e  %1.2e  %1.16e  %5f  %2d        %2.2f%%      %2.2f%%\n", iter, rnorm, rnorm_adj, objective, obj_loss, theta_regul, class_regul, gnorm, stepsize, ls_iter, accur_train, accur_val);
            fprintf(optimfile,"%3d  %1.8e  %1.8e  %1.14e  %1.4e  %1.4e  %1.4e  %1.14e  %5f  %2d        %2.2f%%       %2.2f%%\n", iter, rnorm, rnorm_adj, objective, obj_loss, theta_regul, class_regul, gnorm, stepsize, ls_iter, accur_train, accur_val);
            fflush(optimfile);
        }

        // /* Print to file */
        // char designoptfile[128];
        // char gradientfile[128];
        // sprintf(designoptfile, "design.dat.%d", iter);
        // sprintf(gradientfile, "gradient.dat.%d", iter);
        // write_data(designoptfile, global_design, ndesign);
        // write_data(gradientfile, global_gradient, ndesign);

        /* Check optimization convergence */
        if (  gnorm < gtol )
        {
            if (myid == MASTER_NODE) 
            {
                printf("Optimization has converged. \n");
                printf("Be happy and go home!       \n");
            }
            break;
        }
        
        /* --- Design update --- */


        /* Compute new design on first processor */
        if (myid == MASTER_NODE)
        {
            /* Update the L-BFGS memory */
            if (iter > 0) 
            {
                hessian->update_memory(iter, global_design, global_design0, global_gradient, global_gradient0);
            }

            /* Compute descent direction using L-BFGS Hessian approximation */
            hessian->compute_step(iter, global_gradient, descentdir);

            /* Compute Wolfe condition */
            wolfe = getWolfe(ndesign, global_gradient, descentdir);

            /* Store current design and gradient into *0 vectors */
            copy_vector(ndesign, global_design, global_design0);
            copy_vector(ndesign, global_gradient, global_gradient0);

            /* Update the global design using the initial stepsize */
            update_design(ndesign, stepsize, descentdir, global_design);

            /* Pass the design to the app */
            split_into_4vectors(global_design, ntheta_open, app->theta_open, ntheta, app->theta, nclassW, app->classW, nclasses, app->classMu);

        }
        /* Communicate the new design to all processors */
        MPI_Bcast(app->theta_open, ntheta_open, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
        MPI_Bcast(app->theta,      ntheta,      MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
        MPI_Bcast(app->classW,     nclassW,     MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
        MPI_Bcast(app->classMu,    nclasses,    MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);

        /* Communicate wolfe condition */
        MPI_Bcast(&wolfe, 1, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);

        /* --- Backtracking linesearch --- */
        stepsize = stepsize_init;
        for (ls_iter = 0; ls_iter < ls_maxiter; ls_iter++)
        {
            /* Compute new objective function value for current trial step */
            braid_SetPrintLevel(core_train, 0);
            braid_SetObjectiveOnly(core_train, 1);
            app->training = 1;
            braid_Drive(core_train);
            braid_GetObjective(core_train, &ls_objective);

            if (myid == MASTER_NODE) printf("ls_iter %d: ls_objective %1.14e\n", ls_iter, ls_objective);

            /* Test the wolfe condition */
            if (ls_objective <= objective - ls_param * stepsize * wolfe ) 
            {
                /* Success, use this new design */
                break;
            }
            else
            {
                /* Test for line-search failure */
                if (ls_iter == ls_maxiter - 1)
                {
                    if (myid == MASTER_NODE) printf("\n\n   WARNING: LINESEARCH FAILED! \n\n");
                    break;
                }

                /* Decrease the stepsize */
                stepsize = stepsize * ls_factor;

                /* Compute new design on first processor */
                if (myid == MASTER_NODE)
                {
                    /* Restore the old design */
                    copy_vector(ndesign, global_design0, global_design);

                    /* Update the design with new stepsize */
                    update_design(ndesign, stepsize, descentdir, global_design);
                    split_into_4vectors(global_design, ntheta_open, app->theta_open, ntheta, app->theta, nclassW, app->classW, nclasses, app->classMu);
                }

                /* Communicate the new design to all processors */
                MPI_Bcast(app->theta_open, ntheta_open, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
                MPI_Bcast(app->theta,      ntheta,      MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
                MPI_Bcast(app->classW,     nclassW,     MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
                MPI_Bcast(app->classMu,    nclasses,    MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
            }

        }

        /* Print memory consumption and time in each optimization iteration */
        // getrusage(RUSAGE_SELF,&r_usage);
        // printf(" sys: %d memory  %.2f MB\n", myid, (double) r_usage.ru_maxrss / 1024.0);

   }

    /* --- Run a final propagation ---- */

    /* Parallel-in-layer propagation and gradient computation  */
    braid_SetObjectiveOnly(core_train, 0);
    app->training = 1;
    braid_Drive(core_train);

        /* On Masternode: Get gradient data from app and put it into global_gradient */
    igrad = 0;
    MPI_Reduce(app->theta_open_grad, &(global_gradient[igrad]), ntheta_open, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
    igrad += ntheta_open;
    MPI_Reduce(app->theta_grad, &(global_gradient[igrad]), ntheta, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
    igrad += ntheta;
    MPI_Reduce(app->classW_grad, &(global_gradient[igrad]), nclassW, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
    igrad += nclassW;
    MPI_Reduce(app->classMu_grad, &(global_gradient[igrad]), nclasses, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);

    /* Compute gradient norm */
    mygnorm = 0.0;
    if (myid == MASTER_NODE) {
        mygnorm = vector_normsq(ndesign, global_gradient);
    } 
    MPI_Allreduce(&mygnorm, &gnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Get objective function value and prediction accuracy for training data */
    braid_GetObjective(core_train, &objective);
    MPI_Allreduce(&app->loss, &obj_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&app->accuracy, &accur_train, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    
    /* --- Output --- */
    if (myid == MASTER_NODE)
    {
        printf("\n Loss          %1.14e", obj_loss);
        printf("\n Objective     %1.14e", objective);
        printf("\n Gradientnorm: %1.14e", gnorm);
        printf("\n\n");


        /* Print to file */
        write_data("design_opt.dat", global_design, ndesign);
        write_data("gradient.dat", global_gradient, ndesign);
    }



    /* Print time and memory consumption */
    StopTime = MPI_Wtime();
    UsedTime = StopTime-StartTime;
    getrusage(RUSAGE_SELF,&r_usage);
    myMB = (double) r_usage.ru_maxrss / 1024.0;
    MPI_Allreduce(&myMB, &globalMB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // printf("%d; Memory Usage: %.2f MB\n",myid, myMB);
    if (myid == MASTER_NODE)
    {
        printf("\n");
        printf(" Used Time:        %.2f seconds\n",UsedTime);
        printf(" Global Memory:    %.2f MB\n", globalMB);
        printf(" Processors used:  %d\n", size);
        printf("\n");
    }


    /* Clean up */
    if (myid == MASTER_NODE)
    {
        delete [] Ytrain;
        delete [] Yval;
    }
    if (myid == size -1)
    {
        delete [] Ctrain;
        delete [] Cval;
    }

    if (myid == MASTER_NODE)
    {
        delete hessian;
        delete [] global_design;
        delete [] global_design0;
        delete [] global_gradient;
        delete [] global_gradient0;
        delete [] descentdir;
    }
    delete [] theta;
    delete [] theta_grad;
    delete [] theta_open;
    delete [] theta_open_grad;
    delete [] classW;
    delete [] classW_grad;
    delete [] classMu;
    delete [] classMu_grad;

    delete layer;
    delete openlayer;

    app->training = 1;
    braid_Destroy(core_train);
    app->training = 0;
    braid_Destroy(core_val);

    free(app);
    MPI_Finalize();

    if (myid == MASTER_NODE)
    {
        fclose(optimfile);
        printf("Optimfile: %s\n", optimfilename);
    }


    return 0;
}
