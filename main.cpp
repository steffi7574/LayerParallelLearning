#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

// #include "lib.hpp"
#include "hessianApprox.hpp"
#include "util.hpp"
#include "layer.hpp"
#include "braid.h"
#include "_braid.h"
#include "braid_wrapper.hpp"
#include "parser.h"
#include "network.hpp"

#define MASTER_NODE 0
#define USE_BFGS  1
#define USE_LBFGS 2
#define USE_IDENTITY  3


int main (int argc, char *argv[])
{
    /* --- Data set --- */
    int      ntraining;               /**< Number of elements in training data */
    int      nvalidation;             /**< Number of elements in validation data */
    int      nfeatures;               /**< Number of features in the data set */
    int      nclasses;                /**< Number of classes / Clabels */
    double **train_examples = NULL;   /**< Traning examples */
    double **train_labels   = NULL;   /**< Training labels*/
    double **val_examples   = NULL;   /**< Validation examples */
    double **val_labels     = NULL;   /**< Validation labels*/
    /* --- Network --- */
    int      nlayers;                  /**< Number of layers / time steps */
    int      nchannels;               /**< Number of channels of the netword (width) */
    double   T;                       /**< Final time */
    int      activation;              /**< Enumerator for the activation function */
    Network *network;                 /**< DNN Network architecture */
    /* --- Optimization --- */
    int      ndesign;             /**< Number of local design variables on this processor */
    int      ndesign_global;      /**< Number of global design variables (sum of local)*/
    // double  *design;              /**< Pointer to global design vector */
    // double  *design0;             /**< Old design at last iteration */
    // double  *gradient;            /**< Pointer to global gradient vector */
    // double  *gradient0;           /**< Old gradient at last iteration*/
    double  *descentdir;          /**< Descent direction for design updates */
    double   objective;           /**< Optimization objective */
    double   wolfe;               /**< Holding the wolfe condition value */
    double   gamma_tik;           /**< Parameter for Tikhonov regularization of the weights and bias*/
    double   gamma_ddt;           /**< Parameter for time-derivative regularization of the weights and bias */
    double   gamma_class;         /**< Parameter for regularization of classification weights and bias*/
    double   weights_open_init;   /**< Factor for scaling initial opening layer weights and biases */
    double   weights_init;        /**< Factor for scaling initial weights and bias of intermediate layers*/
    double   weights_class_init;  /**< Factor for scaling initial classification weights and biases */
    double   stepsize_init;       /**< Initial stepsize for design updates */
    int      maxoptimiter;        /**< Maximum number of optimization iterations */
    double   rnorm;               /**< Space-time Norm of the state variables */
    double   rnorm_adj;           /**< Space-time norm of the adjoint variables */
    double   gnorm;               /**< Norm of the gradient */
    double   gtol;                /**< Stoping tolerance on the gradient norm */
    int      ls_maxiter;          /**< Max. number of linesearch iterations */
    double   ls_factor;           /**< Reduction factor for linesearch */
    double   ls_param;            /**< Parameter in wolfe condition test */
    int      hessian_approx;      /**< Hessian approximation (USE_BFGS or L-BFGS) */
    int      lbfgs_stages;        /**< Number of stages of L-bfgs method */
    /* --- PinT --- */
    braid_Core core_train;      /**< Braid core for training data */
    braid_Core core_val;        /**< Braid core for validation data */
    braid_Core core_adj;        /**< Braid core for adjoint computation */
    my_App  *app_train;         /**< Braid app for training data */
    my_App  *app_val;           /**< Braid app for validation data */
    int      myid;              /**< Processor rank */
    int      size;              /**< Number of processors */
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

    braid_BaseVector ubase;
    double accur_train, loss_train;

    struct rusage r_usage;
    double StartTime, StopTime, UsedTime, myMB, globalMB; 
    char  optimfilename[255];
    FILE *optimfile;   
    char* activname;
    char *datafolder, *ftrain_ex, *fval_ex, *ftrain_labels, *fval_labels;
    double mygnorm, stepsize, ls_objective;
    int nreq, ls_iter;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    

    /* --- Set DEFAULT parameters of the config file options --- */ 

    ntraining          = 5000;
    nvalidation        = 200;
    nfeatures          = 2;
    nclasses           = 5;
    nchannels          = 8;
    nlayers            = 32;
    T                  = 10.0;
    activation         = Layer::RELU;
    braid_cfactor      = 4;
    braid_maxlevels    = 10;
    braid_maxiter      = 3;
    braid_abstol       = 1e-10;
    braid_abstoladj    = 1e-06;
    braid_printlevel   = 1;
    braid_accesslevel  = 0;
    braid_setskip      = 0;
    braid_fmg          = 0;
    braid_nrelax       = 1;
    gamma_tik          = 1e-07;
    gamma_ddt          = 1e-07;
    gamma_class        = 1e-07;
    stepsize_init      = 1.0;
    maxoptimiter       = 500;
    gtol               = 1e-08;
    ls_maxiter         = 20;
    ls_factor          = 0.5;
    weights_open_init  = 0.001;
    weights_init       = 0.0;
    weights_class_init = 0.001;
    hessian_approx     = USE_LBFGS;
    lbfgs_stages       = 20;


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

        if ( strcmp(co->key, "datafolder") == 0 )
        {
            datafolder = co->value;
        }
        else if ( strcmp(co->key, "ftrain_ex") == 0 )
        {
            ftrain_ex = co->value;
        }
        else if ( strcmp(co->key, "ftrain_labels") == 0 )
        {
            ftrain_labels = co->value;
        }
        else if ( strcmp(co->key, "fval_ex") == 0 )
        {
            fval_ex = co->value;
        }
        else if ( strcmp(co->key, "fval_labels") == 0 )
        {
            fval_labels = co->value;
        }
        else if ( strcmp(co->key, "ntraining") == 0 )
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
            if (strcmp(co->value, "tanh") == 0 )
            {
                activation = Layer::TANH;
                activname  = "tanh";
            }
            else if ( strcmp(co->value, "ReLu") == 0 )
            {
                activation = Layer::RELU;
                activname  = "ReLu";
            }
            else if (strcmp(co->value, "SmoothReLu") == 0 )
            {
                activation = Layer::SMRELU;
                activname  = "SmoothRelu";
            }
            else
            {
                printf("Invalid activation function!");
                MPI_Finalize();
                return(0);
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
        else if ( strcmp(co->key, "gamma_tik") == 0 )
        {
            gamma_tik = atof(co->value);
        }
        else if ( strcmp(co->key, "gamma_ddt") == 0 )
        {
            gamma_ddt = atof(co->value);
        }
        else if ( strcmp(co->key, "gamma_class") == 0 )
        {
            gamma_class= atof(co->value);
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
        else if ( strcmp(co->key, "weights_open_init") == 0 )
        {
           weights_open_init = atof(co->value);
        }
        else if ( strcmp(co->key, "weights_init") == 0 )
        {
           weights_init = atof(co->value);
        }
        else if ( strcmp(co->key, "weights_class_init") == 0 )
        {
           weights_class_init = atof(co->value);
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
            else if (strcmp(co->value, "Identity") == 0 )
            {
                hessian_approx = USE_IDENTITY;
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

    // nlayers = nlayers -1;

    /*--- INITIALIZATION ---*/

    /* Set the data file names */
    char train_ex_filename[255];
    char train_lab_filename[255];
    char val_ex_filename[255];
    char val_lab_filename[255];
    sprintf(train_ex_filename,  "%s/%s", datafolder, ftrain_ex);
    sprintf(train_lab_filename, "%s/%s", datafolder, ftrain_labels);
    sprintf(val_ex_filename,    "%s/%s", datafolder, fval_ex);
    sprintf(val_lab_filename,   "%s/%s", datafolder, fval_labels);

    /* Read training data */
    train_examples = new double* [ntraining];
    train_labels   = new double* [ntraining];
    for (int ix = 0; ix<ntraining; ix++)
    {
        train_examples[ix] = new double[nfeatures];
        train_labels[ix]   = new double[nclasses];
    }
    read_matrix(train_ex_filename,  train_examples, ntraining, nfeatures);
    read_matrix(train_lab_filename, train_labels,   ntraining, nclasses);

    /* Read validation data */
    val_examples = new double* [nvalidation];
    val_labels   = new double* [nvalidation];
    for (int ix = 0; ix<nvalidation; ix++)
    {
        val_examples[ix] = new double[nfeatures];
        val_labels[ix]   = new double[nclasses];
    }
    read_matrix(val_ex_filename,  val_examples, nvalidation, nfeatures);
    read_matrix(val_lab_filename, val_labels,   nvalidation, nclasses);


    /* Create a network */
    network = new Network(nlayers+1, nchannels, T/(double)nlayers, gamma_tik, gamma_ddt, gamma_class);

    /* Initialize xbraid's app structure */
    app_train = (my_App *) malloc(sizeof(my_App));
    app_train->myid        = myid;
    app_train->network     = network;
    app_train->nexamples   = ntraining;
    app_train->examples    = train_examples;
    app_train->labels      = train_labels;
    /* Initialize xbraid's app structure */
    app_val = (my_App *) malloc(sizeof(my_App));
    app_val->myid          = myid;
    app_val->network       = network;
    app_val->nexamples     = nvalidation;
    app_val->examples      = val_examples;
    app_val->labels        = val_labels;


    /* Initializze XBraid for training data */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, nlayers, app_train, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_train);
    /* Init adjoint core for training data */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, nlayers, app_train, my_Step_Adj, my_Init_Adj, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize_Adj, my_BufPack_Adj, my_BufUnpack_Adj, &core_adj);
    braid_SetRevertedRanks(core_adj, 1);

    /* Init XBraid for validation data */
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, T, nlayers, app_val, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_val);


    /* Store primal core in the app */
    app_train->primalcore = core_train;
    app_val->primalcore   = core_val;


    /* Store all points for primal and adjoint */
    braid_SetStorage(core_train, 0);
    braid_SetStorage(core_adj, 0);

    /* Set Braid parameters */
    braid_SetMaxLevels(core_train, braid_maxlevels);
    braid_SetMaxLevels(core_val,   braid_maxlevels);
    braid_SetMaxLevels(core_adj,   braid_maxlevels);
    braid_SetPrintLevel( core_train, braid_printlevel);
    braid_SetPrintLevel( core_val,   braid_printlevel);
    braid_SetPrintLevel( core_adj,   braid_printlevel);
    braid_SetCFactor(core_train, -1, braid_cfactor);
    braid_SetCFactor(core_val,   -1, braid_cfactor);
    braid_SetCFactor(core_adj,   -1, braid_cfactor);
    braid_SetAccessLevel(core_train, braid_accesslevel);
    braid_SetAccessLevel(core_val,   braid_accesslevel);
    braid_SetAccessLevel(core_adj,   braid_accesslevel);
    braid_SetMaxIter(core_train, braid_maxiter);
    braid_SetMaxIter(core_val,   braid_maxiter);
    braid_SetMaxIter(core_adj,   braid_maxiter);
    braid_SetSkip(core_train, braid_setskip);
    braid_SetSkip(core_val,   braid_setskip);
    braid_SetSkip(core_adj,   braid_setskip);
    if (braid_fmg){
        braid_SetFMG(core_train);
        braid_SetFMG(core_val);
        braid_SetFMG(core_adj);
    }
    braid_SetNRelax(core_train, -1, braid_nrelax);
    braid_SetNRelax(core_val,   -1, braid_nrelax);
    braid_SetNRelax(core_adj,   -1, braid_nrelax);
    braid_SetAbsTol(core_train, braid_abstol);
    braid_SetAbsTol(core_val,   braid_abstol);
    braid_SetAbsTol(core_adj,   braid_abstol);

    /* Get xbraid's grid distribution */
    int ilower, iupper;
    _braid_GetDistribution(core_train, &ilower, &iupper);
    printf("%d: %d %d\n", myid, ilower, iupper);

    /* Allocate and initialize the network layers (local design storage) */
    network->createLayers(ilower, iupper, nfeatures, nclasses, activation, weights_init, weights_open_init, weights_class_init);
    ndesign  = network->getnDesign();
    MPI_Allreduce(&ndesign, &ndesign_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    

    /* Initialize hessian approximation on first processor */
    HessianApprox  *hessian;
    if (myid == MASTER_NODE)
    {
        switch (hessian_approx)
        {
            case USE_BFGS:
                hessian = new BFGS(ndesign);
                break;
            case USE_LBFGS: 
                hessian = new L_BFGS(ndesign, lbfgs_stages);
                break;
            case USE_IDENTITY:
                hessian = new Identity(ndesign);
        }

    }


    /* Allocate other optimization vars on first processor */
    if (myid == MASTER_NODE)
    {
        // design0    = new double [ndesign];
        // gradient0  = new double [ndesign];
        descentdir = new double [ndesign_global];
    }

    /* Initialize optimization parameters */
    ls_param    = 1e-4;
    ls_iter     = 0;
    gnorm       = 0.0;
    objective   = 0.0;
    rnorm       = 0.0;
    rnorm_adj   = 0.0;
    stepsize    = stepsize_init;

    /* Open and prepare optimization output file*/
    if (myid == MASTER_NODE)
    {
        sprintf(optimfilename, "%s.dat", "optim");
        optimfile = fopen(optimfilename, "w");
        fprintf(optimfile, "# Problem setup: ntraining           %d \n", ntraining);
        fprintf(optimfile, "#                nvalidation         %d \n", nvalidation);
        fprintf(optimfile, "#                nfeatures           %d \n", nfeatures);
        fprintf(optimfile, "#                nclasses            %d \n", nclasses);
        fprintf(optimfile, "#                nchannels           %d \n", nchannels);
        fprintf(optimfile, "#                nlayers             %d \n", nlayers);
        fprintf(optimfile, "#                T                   %f \n", T);
        fprintf(optimfile, "#                Activation          %s \n", activname);
        fprintf(optimfile, "# XBraid setup:  max levels          %d \n", braid_maxlevels);
        fprintf(optimfile, "#                coasening           %d \n", braid_cfactor);
        fprintf(optimfile, "#                max. braid iter     %d \n", braid_maxiter);
        fprintf(optimfile, "#                abs. tol            %1.e \n", braid_abstol);
        fprintf(optimfile, "#                abs. toladj         %1.e \n", braid_abstoladj);
        fprintf(optimfile, "#                print level         %d \n", braid_printlevel);
        fprintf(optimfile, "#                access level        %d \n", braid_accesslevel);
        fprintf(optimfile, "#                skip?               %d \n", braid_setskip);
        fprintf(optimfile, "#                fmg?                %d \n", braid_fmg);
        fprintf(optimfile, "#                nrelax              %d \n", braid_nrelax);
        fprintf(optimfile, "# Optimization:  gamma_tik           %1.e \n", gamma_tik);
        fprintf(optimfile, "#                gamma_ddt           %1.e \n", gamma_ddt);
        fprintf(optimfile, "#                gamma_class         %1.e \n", gamma_class);
        fprintf(optimfile, "#                stepsize            %f \n", stepsize_init);
        fprintf(optimfile, "#                max. optim iter     %d \n", maxoptimiter);
        fprintf(optimfile, "#                gtol                %1.e \n", gtol);
        fprintf(optimfile, "#                max. ls iter        %d \n", ls_maxiter);
        fprintf(optimfile, "#                ls factor           %f \n", ls_factor);
        fprintf(optimfile, "#                weights_init        %f \n", weights_init);
        fprintf(optimfile, "#                weights_open_init   %f \n", weights_open_init);
        fprintf(optimfile, "#                weights_class_init  %f \n", weights_class_init) ;
        fprintf(optimfile, "#                hessian_approx      %d \n", hessian_approx);
        fprintf(optimfile, "#                lbfgs_stages        %d \n", lbfgs_stages);
        fprintf(optimfile, "\n");
    }

    /* Prepare optimization output */
    if (myid == MASTER_NODE)
    {
       /* Screen output */
       printf("\n#    || r ||          || r_adj ||       Objective             Loss                || grad ||             Stepsize  ls_iter   Accur_train  Accur_val    Time(sec)\n");
       
       fprintf(optimfile, "#    || r ||          || r_adj ||      Objective             Loss                  || grad ||            Stepsize  ls_iter   Accur_train  Accur_val   Time(sec)\n");
    }


    /* --- OPTIMIZATION --- */
    StartTime = MPI_Wtime();
    StopTime  = 0.0;
    for (int iter = 0; iter < maxoptimiter; iter++)
    {

        // /*  Perturb design */
        // int idx =0;
        // design[idx] += 1e-7;

        /* --- Training data: Get objective and compute gradient ---*/ 

        /* Solve with braid */
        braid_SetPrintLevel(core_train, 1);
        braid_Drive(core_train);

        /* Get braid residual norm */
        nreq = -1;
        braid_GetRNorms(core_train, &nreq, &rnorm);
        
        /* Evaluat objective function (loop over every time point) */
        objective = 0.0;
        for (int ilayer = 0; ilayer <= nlayers; ilayer++)
        {
            /* Get braid vector at this time step */
            _braid_UGetVectorRef(core_train, 0, ilayer, &ubase);

            if (ubase != NULL) // this is only true on one processor (the one that stores u)
            {
                objective += evalObjectiveT(app_train, ubase->userVector, ilayer, &loss_train, &accur_train);
            }
        }
        /* Collect objective function for all processors */
        double myobjective = objective;
        MPI_Allreduce(&myobjective, &objective, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        printf("%d: Objective %1.14e Loss %1.14e Accuracy %1.14e\n", myid, objective, loss_train, accur_train);



        printf("\n\n SOLVE ADJOINT WITH XBRAID\n\n");


        /* RUN */
        braid_Drive(core_adj);

        /* Get adjoint residual norms */
        braid_GetRNormAdjoint(core_adj, &rnorm_adj);

        /* Get the gradient */
        printf("%d: local ndesign %d out of %d\n", myid, ndesign, ndesign_global);
        MPI_Gather(network->getDesign(), ndesign, MPI_DOUBLE, descentdir, ndesign, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
        if (myid == MASTER_NODE) write_vector("gradient.dat", descentdir, ndesign_global);

        // /* Reduce gradient on MASTER_NODE (in-place communication)*/
        // if (myid == MASTER_NODE) 
        // {
        //     MPI_Reduce(MPI_IN_PLACE, gradient, ndesign, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
        // }
        // else
        // {
        //     MPI_Reduce(gradient, gradient, ndesign, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);
        // }

        /* --- Validation data: Get accuracy --- */

        // braid_SetObjectiveOnly(core_val, 1);
        // braid_SetPrintLevel( core_val,   0);
        // braid_Drive(core_val);


        // /* --- Optimization control and output ---*/

        // /* Compute and communicate the norm */
        // mygnorm = 0.0;
        // if (myid == MASTER_NODE) {
        //     mygnorm = vec_normsq(ndesign, gradient);
        // } 
        // MPI_Allreduce(&mygnorm, &gnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // gnorm = sqrt(gnorm);

        // /* Communicate loss and accuracy. This is actually not needed, except for printing output. Remove it. */
        // double train_loss, train_accur, val_accur;
        // MPI_Allreduce(&app_train->loss, &train_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // MPI_Allreduce(&app_train->accuracy, &train_accur, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // MPI_Allreduce(&app_val->accuracy, &val_accur, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // /* Output */
        // if (myid == MASTER_NODE)
        // {
   
        //     printf("%3d  %1.8e  %1.8e  %1.14e  %1.14e  %1.14e  %5f  %2d        %2.2f%%      %2.2f%%    %.1f\n", iter, rnorm, rnorm_adj, objective, train_loss, gnorm, stepsize, ls_iter, train_accur, val_accur, UsedTime);
        //     fprintf(optimfile,"%3d  %1.8e  %1.8e  %1.14e  %1.14e  %1.14e  %5f  %2d        %2.2f%%      %2.2f%%     %.1f\n", iter, rnorm, rnorm_adj, objective, train_loss, gnorm, stepsize, ls_iter, train_accur, val_accur, UsedTime);
        //     fflush(optimfile);
        // }

        // /* Check optimization convergence */
        // if (  gnorm < gtol )
        // {
        //     if (myid == MASTER_NODE) 
        //     {
        //         printf("Optimization has converged. \n");
        //         printf("Be happy and go home!       \n");
        //     }
        //     break;
        // }


        // /* --- Design update --- */

        // stepsize = stepsize_init;
        // /* Compute search direction on first processor */
        // if (myid == MASTER_NODE)
        // {
        //     /* Update the L-BFGS memory */
        //     if (iter > 0) 
        //     {
        //         hessian->updateMemory(iter, design, design0, gradient, gradient0);
        //     }
        //     /* Compute descent direction */
        //     hessian->computeDescentDir(iter, gradient, descentdir);

        //     /* Store design and gradient into *0 vectors */
        //     vec_copy(ndesign, design, design0);
        //     vec_copy(ndesign, gradient, gradient0);

        //     /* Compute wolfe condition */
        //     wolfe = vecdot(ndesign, gradient, descentdir);

        //     /* Update the global design using the initial stepsize */
        //     for (int id = 0; id < ndesign; id++)
        //     {
        //         design[id] -= stepsize * descentdir[id];
        //     }
        // }

        // /* Broadcast the new design and wolfe condition to all processors */
        // MPI_Bcast(design, ndesign, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
        // MPI_Bcast(&wolfe, 1, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);

        // /* --- Backtracking linesearch --- */
        // for (ls_iter = 0; ls_iter < ls_maxiter; ls_iter++)
        // {
        //     /* Compute new objective function value for current trial step */
        //     braid_SetPrintLevel(core_train, 0);
        //     braid_SetObjectiveOnly(core_train, 1);
        //     braid_Drive(core_train);
        //     braid_GetObjective(core_train, &ls_objective);

        //     double test = objective - ls_param * stepsize * wolfe;
        //     if (myid == MASTER_NODE) printf("ls_iter %d: %1.14e %1.14e\n", ls_iter, ls_objective, test);
        //     /* Test the wolfe condition */
        //     if (ls_objective <= objective - ls_param * stepsize * wolfe ) 
        //     {
        //         /* Success, use this new design */
        //         break;
        //     }
        //     else
        //     {
        //         /* Test for line-search failure */
        //         if (ls_iter == ls_maxiter - 1)
        //         {
        //             if (myid == MASTER_NODE) printf("\n\n   WARNING: LINESEARCH FAILED! \n\n");
        //             break;
        //         }

        //         /* Decrease the stepsize */
        //         stepsize = stepsize * ls_factor;

        //         /* Compute new design using new stepsize */
        //         if (myid == MASTER_NODE)
        //         {
        //             /* Go back a portion of the step */
        //             for (int id = 0; id < ndesign; id++)
        //             {
        //                 design[id] += stepsize * descentdir[id];
        //             }
        //         }
        //         MPI_Bcast(design, ndesign, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
 
        //     }
 
        // }
 
        /* Print some statistics */
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
            printf("\n");
        }
    }




/** ==================================================================================
 * Adjoint dot test xbarTxdot = ybarTydot
 * where xbar = (dfdx)T ybar
 *       ydot = (dfdx)  xdot
 * choosing xdot to be a vector of all ones, ybar = 1.0;
 * ==================================================================================*/
    // if (size == 1)
    // {
    //      printf("\n\n ============================ \n");
    //      printf(" Adjoint dot test: \n\n");

    //     // read_vector("design.dat", design, ndesign);
         
    //     /* Propagate through braid */
    //     braid_SetObjectiveOnly(core_train, 0);
    //     braid_Drive(core_train);
    //     braid_GetObjective(core_train, &objective);
    //     write_vector("gradient.dat", gradient, ndesign);
    //     double obj0 = objective;

    //     /* Sum up xtx */
    //     double xtx = 0.0;
    //     for (int i = 0; i < ndesign; i++)
    //     {
    //         xtx += gradient[i];
    //     }

    //     /* perturb into direction "only ones" */
    //     double EPS = 1e-7;
    //     for (int i = 0; i < ndesign; i++)
    //     {
    //         design[i] += EPS;
    //     }

    //     /* New objective function evaluation */
    //     braid_SetObjectiveOnly(core_train, 1);
    //     braid_Drive(core_train);
    //     braid_GetObjective(core_train, &objective);
    //     double obj1 = objective;

    //     /* Finite differences */
    //     double yty = (obj1 - obj0)/EPS;


    //     /* Print adjoint dot test result */
    //     printf(" Dot-test: %1.16e  %1.16e\n\n Rel. error  %3.6f %%\n\n", xtx, yty, (yty-xtx)/xtx * 100.);

    // }

/** =======================================
 * Full finite differences 
 * ======================================= */

    // double* findiff = new double[ndesign];
    // double* relerr = new double[ndesign];
    // double errnorm = 0.0;
    // double obj0, obj1, design_store;
    // double EPS;

    // printf("\n--------------------------------\n");
    // printf(" FINITE DIFFERENCE TESTING\n\n");

    // /* Compute baseline objective */
    // // read_vector("design.dat", design, ndesign);
    // braid_SetObjectiveOnly(core_train, 0);
    // braid_Drive(core_train);
    // braid_GetObjective(core_train, &objective);
    // obj0 = objective;

    // EPS = 1e-4;
    // for (int i = 0; i < ndesign; i++)
    // // for (int i = 0; i < 22; i++)
    // // int i=21;
    // {
    //     /* Restore design */
    //     // read_vector("design.dat", design, ndesign);
    
    //     /*  Perturb design */
    //     design_store = design[i];
    //     design[i] += EPS;

    //     /* Recompute objective */
    //     _braid_CoreElt(core_train, warm_restart) = 0;
    //     braid_SetObjectiveOnly(core_train, 1);
    //     braid_SetPrintLevel(core_train, 0);
    //     braid_Drive(core_train);
    //     braid_GetObjective(core_train, &objective);
    //     obj1 = objective;

    //     /* Findiff */
    //     findiff[i] = (obj1 - obj0) / EPS;
    //     relerr[i]  = (gradient[i] - findiff[i]) / findiff[i];
    //     errnorm += pow(relerr[i],2);

    //     printf("\n %4d: % 1.14e % 1.14e, error: % 2.4f",i, findiff[i], gradient[i], relerr[i] * 100.0);

    //     /* Restore design */
    //     design[i] = design_store;
    // }
    // errnorm = sqrt(errnorm);
    // printf("\n FinDiff ErrNorm  %1.14e\n", errnorm);

    // write_vector("findiff.dat", findiff, ndesign); 
    // write_vector("relerr.dat", relerr, ndesign); 
     

 /* ======================================= 
  * check network implementation 
  * ======================================= */
    // network->applyFWD(ntraining, train_examples, train_labels);
    // double accur = network->getAccuracy();
    // double regul = network->evalRegularization();
    // objective = network->getLoss() + regul;
    // printf("\n --- \n");
    // printf(" Network: obj %1.14e \n", objective);
    // printf(" ---\n");

    /* Print some statistics */
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
    delete network;
    braid_Destroy(core_train);
    braid_Destroy(core_val);
    free(app_train);
    free(app_val);

    if (myid == MASTER_NODE)
    {
        delete hessian;
        // delete [] design0;
        // delete [] gradient0;
        delete [] descentdir;
    }

    for (int ix = 0; ix<ntraining; ix++)
    {
        delete [] train_examples[ix];
        delete [] train_labels[ix];
    }
    delete [] train_examples;
    delete [] train_labels;
    for (int ix = 0; ix<nvalidation; ix++)
    {
        delete [] val_examples[ix];
        delete [] val_labels[ix];
    }
    delete [] val_examples;
    delete [] val_labels;

    if (myid == MASTER_NODE)
    {
        fclose(optimfile);
        printf("Optimfile: %s\n", optimfilename);
    }

    MPI_Finalize();
    return 0;
}
