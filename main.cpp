#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "defs.hpp"
#include "hessianApprox.hpp"
#include "util.hpp"
#include "layer.hpp"
#include "braid_wrapper.hpp"
#include "config.hpp"
#include "network.hpp"
#include "dataset.hpp"

#define MASTER_NODE 0

int main (int argc, char *argv[])
{
    /* --- Data set --- */
    DataSet *trainingdata;
    DataSet *validationdata;
    /* --- Network --- */
    int      nhiddenlayers;           /**< Number of hidden layers = number of xbraid steps */
    Network *network;                 /**< DNN Network architecture */
    /* --- Optimization --- */
    int      ndesign_local;             /**< Number of local design variables on this processor */
    int      ndesign_layermax;          /**< Max. number of design variables over all hidden layers */
    int      ndesign_global;      /**< Number of global design variables (sum of local)*/
    MyReal  *design_init=0;       /**< Temporary vector for initializing the design (on P0) */
    MyReal  *ascentdir=0;        /**< Direction for design updates */
    MyReal   objective;           /**< Optimization objective */
    MyReal   wolfe;               /**< Holding the wolfe condition value */
    MyReal   rnorm;               /**< Space-time Norm of the state variables */
    MyReal   rnorm_adj;           /**< Space-time norm of the adjoint variables */
    MyReal   gnorm;               /**< Norm of the gradient */
    MyReal   ls_param;            /**< Parameter in wolfe condition test */
    /* --- PinT --- */
    braid_Core core_train;      /**< Braid core for training data */
    braid_Core core_val;        /**< Braid core for validation data */
    braid_Core core_adj;        /**< Braid core for adjoint computation */
    my_App  *app_train;         /**< Braid app for training data */
    my_App  *app_val;           /**< Braid app for validation data */
    int      myid;              /**< Processor rank */
    int      size;              /**< Number of processors */

    MyReal accur_train = 0.0;   /**< Accuracy on training data */
    MyReal accur_val   = 0.0;   /**< Accuracy on validation data */
    MyReal loss_train  = 0.0;   /**< Loss function on training data */
    MyReal loss_val    = 0.0;   /**< Loss function on validation data */

    int ilower, iupper;         /**< Index of first and last layer stored on this processor */
    struct rusage r_usage;
    MyReal StartTime, StopTime, myMB, globalMB; 
    MyReal UsedTime = 0.0;
    char optimfilename[255];
    char train_ex_filename[255], train_lab_filename[255];
    char val_ex_filename[255], val_lab_filename[255];
    FILE *optimfile = 0;   
    MyReal ls_stepsize, ls_objective, test_obj;
    MyReal stepsize;
    int nreq = -1;
    int ls_iter;
    braid_BaseVector ubase;
    braid_Vector     u;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


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

    /* Read config file */
    Config* config = new Config();
    int err = config->readFromFile(argv[1]);
    if (err)
    {
        printf("\nError while reading config file!\n");
        MPI_Finalize();
        return (0);
    }


    /*--- INITIALIZATION ---*/

    /* Set the data file names */
    sprintf(train_ex_filename,  "%s/%s", config->datafolder, config->ftrain_ex);
    sprintf(train_lab_filename, "%s/%s", config->datafolder, config->ftrain_labels);
    sprintf(val_ex_filename,    "%s/%s", config->datafolder, config->fval_ex);
    sprintf(val_lab_filename,   "%s/%s", config->datafolder, config->fval_labels);

    /* Allocate and read training and validation data */
    trainingdata = new DataSet(size, myid, config->ntraining,   config->nfeatures, config->nclasses, config->nbatch);
    trainingdata->readData(train_ex_filename, train_lab_filename);

    validationdata = new DataSet(size, myid, config->nvalidation, config->nfeatures, config->nclasses, config->nvalidation);  // full validation set!
    validationdata->readData(val_ex_filename, val_lab_filename);


    /* Total number of hidden layers is nlayers minus opening layer minus classification layers) */
    nhiddenlayers = config->nlayers - 2;

    /* Initializze primal and adjoint XBraid for training data */
    app_train = (my_App *) malloc(sizeof(my_App));
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, config->T, nhiddenlayers, app_train, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_train);
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, config->T, nhiddenlayers, app_train, my_Step_Adj, my_Init_Adj, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize_Adj, my_BufPack_Adj, my_BufUnpack_Adj, &core_adj);
    /* Store all primal points */
    braid_SetStorage(core_train, 0);
    /* Revert ranks for solveadjointwithxbraid */
    braid_SetRevertedRanks(core_adj, 1);

    /* Init XBraid for validation data */
    app_val = (my_App *) malloc(sizeof(my_App));
    braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, config->T, nhiddenlayers, app_val, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core_val);

    /* Set all Braid parameters */
    braid_SetConfigOptions(core_train, config);
    braid_SetConfigOptions(core_adj, config);
    braid_SetConfigOptions(core_val, config);

    /* Get xbraid's grid distribution */
    _braid_GetDistribution(core_train, &ilower, &iupper);

    /* Create network and layers */
    network = new Network(ilower, iupper, config);

    /* Get local and global number of design variables. */ 
    ndesign_local          = network->getnDesignLocal();
    int myndesign_layermax = network->getnDesignLayermax();
    MPI_Allreduce(&ndesign_local, &ndesign_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&myndesign_layermax, &ndesign_layermax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    int startid = ilower;
    if (ilower == 0) startid = -1;
    printf("%d: Layer range: [%d, %d] / %d\n", myid, startid, iupper, config->nlayers);
    printf("%d: Design variables (local/global): %d/%d\n", myid, ndesign_local, ndesign_global);

    /* Initialize design with random numbers (do on one processor and scatter for scaling test) */
    if (myid == MASTER_NODE)
    {
        srand(1.0);
        design_init = new MyReal[ndesign_global];
        for (int i = 0; i < ndesign_global; i++)
        {
            design_init[i] = (MyReal) rand() / ((MyReal) RAND_MAX);
        }
    }
    MPI_ScatterVector(design_init, network->getDesign(), ndesign_local, MASTER_NODE, MPI_COMM_WORLD);
    network->initialize(config);
    network->MPI_CommunicateNeighbours(MPI_COMM_WORLD);

    /* Initialize xbraid's app structure */
    app_train->primalcore       = core_train;
    app_train->myid             = myid;
    app_train->network          = network;
    app_train->data             = trainingdata;
    app_train->ndesign_layermax = ndesign_layermax;
    app_val->primalcore       = core_val;
    app_val->myid             = myid;
    app_val->network          = network;
    app_val->data             = validationdata;
    app_val->ndesign_layermax = ndesign_layermax;


    /* Initialize hessian approximation on first processor */
    HessianApprox  *hessian = 0;
    switch (config->hessianapprox_type)
    {
        case BFGS_SERIAL:
            hessian = new BFGS(MPI_COMM_WORLD, ndesign_local);
            break;
        case LBFGS: 
            hessian = new L_BFGS(MPI_COMM_WORLD, ndesign_local, config->lbfgs_stages);
            break;
        case IDENTITY:
            hessian = new Identity(MPI_COMM_WORLD, ndesign_local);
    }

    /* Allocate ascent direction for design updates */
    ascentdir = new MyReal[ndesign_local];

    /* Initialize optimization parameters */
    ls_param    = 1e-4;
    ls_iter     = 0;
    gnorm       = 0.0;
    objective   = 0.0;
    rnorm       = 0.0;
    rnorm_adj   = 0.0;
    stepsize    = config->getStepsize(0);
    ls_stepsize = stepsize;

    /* Open and prepare optimization output file*/
    if (myid == MASTER_NODE)
    {
        sprintf(optimfilename, "%s.dat", "optim");
        optimfile = fopen(optimfilename, "w");
        config->writeToFile(optimfile);

       fprintf(optimfile, "#    || r ||          || r_adj ||      Objective             Loss                  || grad ||            Stepsize  ls_iter   Accur_train  Accur_val   Time(sec)\n");

       /* Screen output */
       printf("\n#    || r ||          || r_adj ||      Objective             Loss                 || grad ||             Stepsize  ls_iter   Accur_train  Accur_val   Time(sec)\n");
    }


#if 1
    /* --- OPTIMIZATION --- */
    StartTime = MPI_Wtime();
    StopTime  = 0.0;
    UsedTime = 0.0;
    for (int iter = 0; iter < config->maxoptimiter; iter++)
    {

        /* --- Training data: Get objective and gradient ---*/ 
        
        /* Set up the current batch */
        trainingdata->selectBatch(config->batch_type);
        // trainingdata->printBatch();

        /* Solve state equation with braid */
        nreq = -1;
        braid_SetPrintLevel(core_train, config->braid_printlevel);
        braid_evalInit(core_train, app_train);
        braid_Drive(core_train);
        braid_evalObjective(core_train, app_train, &objective, &loss_train, &accur_train);
        braid_GetRNorms(core_train, &nreq, &rnorm);

        /* Solve adjoint equation with XBraid */
        nreq = -1;
        braid_SetPrintLevel(core_adj, config->braid_printlevel);
        braid_evalObjectiveDiff(core_adj, app_train);
        braid_Drive(core_adj);
        braid_evalInitDiff(core_adj, app_train);
        braid_GetRNorms(core_adj, &nreq, &rnorm_adj);

        /* --- Validation data: Get accuracy --- */

        if ( config->validationlevel > 0 )
        {
            braid_SetPrintLevel( core_val, 1);
            braid_evalInit(core_val, app_val);
            braid_Drive(core_val);
            /* Get loss and accuracy */
            _braid_UGetLast(core_val, &ubase);
            if (ubase != NULL) // This is true only on last processor
            {
                u = ubase->userVector;
                network->evalClassification(validationdata, u->state, &loss_val, &accur_val, 0);
            }
        }


        /* --- Optimization control and output ---*/

        /* Compute global gradient norm */
        gnorm = vecnorm_par(ndesign_local, network->getGradient(), MPI_COMM_WORLD);

        /* Communicate loss and accuracy. This is actually only needed for output. Remove it. */
        MyReal losstrain_out  = 0.0; 
        MyReal lossval_out    = 0.0; 
        MyReal accurtrain_out = 0.0; 
        MyReal accurval_out   = 0.0; 
        MPI_Allreduce(&loss_train, &losstrain_out, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&loss_val, &lossval_out, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&accur_train, &accurtrain_out, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&accur_val, &accurval_out, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);

        /* Output */
        StopTime = MPI_Wtime();
        UsedTime = StopTime-StartTime;
        if (myid == MASTER_NODE)
        {
            printf("%03d  %1.8e  %1.8e  %1.14e  %1.14e  %1.14e  %5f  %2d        %2.2f%%      %2.2f%%    %.1f\n", iter, rnorm, rnorm_adj, objective, losstrain_out, gnorm, stepsize, ls_iter, accurtrain_out, accurval_out, UsedTime);
            fprintf(optimfile,"%03d  %1.8e  %1.8e  %1.14e  %1.14e  %1.14e  %5f  %2d        %2.2f%%      %2.2f%%     %.1f\n", iter, rnorm, rnorm_adj, objective, losstrain_out, gnorm, stepsize, ls_iter, accurtrain_out, accurval_out, UsedTime);
            fflush(optimfile);
        }

        /* Check optimization convergence */
        if (  gnorm < config->gtol )
        {
            if (myid == MASTER_NODE) 
            {
                printf("Optimization has converged. \n");
                printf("Be happy and go home!       \n");
            }
            break;
        }
        if ( iter == config->maxoptimiter - 1 )
        {
            if (myid == MASTER_NODE)
            {
                printf("\nMax. optimization iterations reached.\n");
            }
            break;
        }


        /* --- Design update --- */

        /* Compute search direction */
        hessian->updateMemory(iter, network->getDesign(), network->getGradient());
        hessian->computeAscentDir(iter, network->getGradient(), ascentdir);
        
        /* Update the design in negative ascent direction */
        stepsize = config->getStepsize(iter);
        network->updateDesign( -1.0 * stepsize, ascentdir, MPI_COMM_WORLD);


        /* --- Backtracking linesearch --- */

        if (config->stepsize_type == BACKTRACKINGLS)
        {
            /* Compute wolfe condition */
            wolfe = vecdot_par(ndesign_local, network->getGradient(), ascentdir, MPI_COMM_WORLD);

            /* Start linesearch iterations */
            ls_stepsize  = config->getStepsize(iter);
            stepsize     = ls_stepsize;
            for (ls_iter = 0; ls_iter < config->ls_maxiter; ls_iter++)
            {
                /* Compute new objective function value for current trial step */
                braid_SetPrintLevel(core_train, 0);
                braid_evalInit(core_train, app_train);
                braid_Drive(core_train);
                braid_evalObjective(core_train, app_train, &ls_objective, &loss_train, &accur_train);

                test_obj = objective - ls_param * ls_stepsize * wolfe;
                if (myid == MASTER_NODE) printf("ls_iter %d: %1.14e %1.14e\n", ls_iter, ls_objective, test_obj);
                /* Test the wolfe condition */
                if (ls_objective <= test_obj) 
                {
                    /* Success, use this new design */
                    break;
                }
                else
                {
                    /* Test for line-search failure */
                    if (ls_iter == config->ls_maxiter - 1)
                    {
                        if (myid == MASTER_NODE) printf("\n\n   WARNING: LINESEARCH FAILED! \n\n");
                        break;
                    }

                    /* Go back part of the step */
                    network->updateDesign((1.0 - config->ls_factor) * stepsize, ascentdir, MPI_COMM_WORLD);

                    /* Decrease the stepsize */
                    ls_stepsize = ls_stepsize * config->ls_factor;
                    stepsize = ls_stepsize;
                }
            }
        }
    }

    /* --- Run final validation and write prediction file --- */
    if (config->validationlevel > -1)
    {
        if (myid == MASTER_NODE) printf("\n --- Run final validation ---\n");
        braid_SetPrintLevel( core_val, 0);
        braid_evalInit(core_val, app_val);
        braid_Drive(core_val);
        /* Get loss and accuracy */
        _braid_UGetLast(core_val, &ubase);
        if (ubase != NULL) // This is only true on last processor 
        {
            u = ubase->userVector;
            network->evalClassification(validationdata, u->state, &loss_val, &accur_val, 1);
            printf("Final validation accuracy:  %2.2f%%\n", accur_val);
        }
    }

    // write_vector("design.dat", design, ndesign);
#endif




/** ==================================================================================
 * Adjoint dot test xbarTxdot = ybarTydot
 * where xbar = (dfdx)T ybar
 *       ydot = (dfdx)  xdot
 * choosing xdot to be a vector of all ones, ybar = 1.0;
 * ==================================================================================*/
#if 0
 
    if (size == 1)
    {
         MyReal obj1, obj0;
        //  int nconv_size = 3;

         printf("\n\n ============================ \n");
         printf(" Adjoint dot test: \n\n");
        //  printf("   ndesign   = %d (calc = %d)\n",ndesign,
        //                                           nchannels*config->nclasses+config->nclasses // class layer
        //                                           +(nlayers-2)+(nlayers-2)*(nconv_size*nconv_size*(nchannels/config->nfeatures)*(nchannels/config->nfeatures))); // con layers
        //  printf("   nchannels = %d\n",nchannels);
        //  printf("   nlayers   = %d\n",nlayers); 
        //  printf("   conv_size = %d\n",nconv_size);
        //  printf("   config->nclasses  = %d\n\n",config->nclasses);


        /* TODO: read some design */

        /* Propagate through braid */ 
        braid_evalInit(core_train, app_train);
        braid_Drive(core_train);
        braid_evalObjective(core_train, app_train, &obj0, &loss_train, &accur_train);

        /* Eval gradient */
        braid_evalObjectiveDiff(core_adj, app_train);
        braid_Drive(core_adj);
        braid_evalInitDiff(core_adj, app_train);


        MyReal xtx = 0.0;
        MyReal EPS = 1e-7;
        for (int i = 0; i < ndesign_global; i++)
        {
            /* Sum up xtx */
            xtx += network->getGradient()[i];
            /* perturb into direction "only ones" */
            network->getDesign()[i] += EPS;
        }


        /* New objective function evaluation */
        braid_evalInit(core_train, app_train);
        braid_Drive(core_train);
        braid_evalObjective(core_train, app_train, &obj1, &loss_train, &accur_train);

        /* Finite differences */
        MyReal yty = (obj1 - obj0)/EPS;


        /* Print adjoint dot test result */
        printf(" Dot-test: %1.16e  %1.16e\n\n Rel. error  %3.6f %%\n\n", xtx, yty, (yty-xtx)/xtx * 100.);
        printf(" obj0 %1.14e, obj1 %1.14e\n", obj0, obj1);

    }

#endif

/** =======================================
 * Full finite differences 
 * ======================================= */

    // MyReal* findiff = new MyReal[ndesign];
    // MyReal* relerr = new MyReal[ndesign];
    // MyReal errnorm = 0.0;
    // MyReal obj0, obj1, design_store;
    // MyReal EPS;

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
    // network->applyFWD(config->ntraining, train_examples, train_labels);
    // MyReal accur = network->getAccuracy();
    // MyReal regul = network->evalRegularization();
    // objective = network->getLoss() + regul;
    // printf("\n --- \n");
    // printf(" Network: obj %1.14e \n", objective);
    // printf(" ---\n");

    /* Print some statistics */
    StopTime = MPI_Wtime();
    UsedTime = StopTime-StartTime;
    getrusage(RUSAGE_SELF,&r_usage);
    myMB = (MyReal) r_usage.ru_maxrss / 1024.0;
    MPI_Allreduce(&myMB, &globalMB, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);

    // printf("%d; Memory Usage: %.2f MB\n",myid, myMB);
    if (myid == MASTER_NODE)
    {
        printf("\n");
        printf(" Used Time:        %.2f seconds\n",UsedTime);
        printf(" Global Memory:    %.2f MB\n", globalMB);
        printf(" Processors used:  %d\n", size);
        printf("\n");
    }


    /* Clean up XBraid */
    delete network;
    braid_Destroy(core_train);
    braid_Destroy(core_adj);
    if (config->validationlevel >= 0) braid_Destroy(core_val);
    free(app_train);
    free(app_val);

    /* Delete optimization vars */
    delete hessian;
    delete [] design_init;
    delete [] ascentdir;

    /* Delete training and validation examples  */
    delete trainingdata;
    delete validationdata;

    /* Close optim file */
    if (myid == MASTER_NODE)
    {
        fclose(optimfile);
        printf("Optimfile: %s\n", optimfilename);
    }

    delete config;

    MPI_Finalize();
    return 0;
}
