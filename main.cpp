#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "defs.hpp"
#include "optimizer.hpp"
#include "util.hpp"
#include "layer.hpp"
#include "braid_wrapper.hpp"
#include "config.hpp"
#include "network.hpp"
#include "dataset.hpp"
#include "opt_wrapper.hpp"

#define MASTER_NODE 0

int main (int argc, char *argv[])
{
    /* --- Data --- */
    Config*  config;              /**< Storing configurations */
    DataSet* trainingdata;        /**< Training dataset */
    DataSet* validationdata;      /**< Validation dataset */

    /* --- Network --- */
    Network* network;             /**< DNN Network architecture */
    int      ilower, iupper;         /**< Index of first and last layer stored on this processor */
    MyReal   accur_train = 0.0;   /**< Accuracy on training data */
    MyReal   accur_val   = 0.0;   /**< Accuracy on validation data */
    MyReal   loss_train  = 0.0;   /**< Loss function on training data */
    MyReal   loss_val    = 0.0;   /**< Loss function on validation data */
    MyReal   losstrain_out  = 0.0; 
    MyReal   lossval_out    = 0.0; 
    MyReal   accurtrain_out = 0.0; 
    MyReal   accurval_out   = 0.0; 
 
    /* --- XBraid --- */
    myBraidApp        *primaltrainapp;   /**< Braid App for training data */
    myAdjointBraidApp *adjointtrainapp;  /**< Adjoint Braid for training data */
    myBraidApp        *primalvalapp;     /**< Braid App for validation data */

    /* --- Optimization --- */
    int     ndesign_local;       /**< Number of local design variables on this processor */
    int     ndesign_global;      /**< Number of global design variables (sum of local)*/
    MyReal* ascentdir=0;         /**< Direction for design updates */
    MyReal  objective;           /**< Optimization objective */
    MyReal  wolfe;               /**< Holding the wolfe condition value */
    MyReal  rnorm;               /**< Space-time Norm of the state variables */
    MyReal  rnorm_adj;           /**< Space-time norm of the adjoint variables */
    MyReal  gnorm;               /**< Norm of the gradient */
    MyReal  ls_param;            /**< Parameter in wolfe condition test */
    MyReal  stepsize;            /**< Stepsize used for design update */
    char    optimfilename[255];  
    FILE    *optimfile = 0;   
    MyReal  ls_stepsize;
    MyReal  ls_objective, test_obj;
    int     ls_iter;


    /* --- other --- */
    int      myid;              
    int      size;
    struct   rusage r_usage;
    MyReal   StartTime, StopTime, myMB, globalMB; 
    MyReal   UsedTime = 0.0;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    /*--- INITIALIZATION ---*/

    /* Instantiate objects */
    config         = new Config();
    trainingdata   = new DataSet();
    validationdata = new DataSet();
    network        = new Network();


    /* Read config file */
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
    int err = config->readFromFile(argv[1]);
    if (err)
    {
        printf("\nError while reading config file!\n");
        MPI_Finalize();
        return (0);
    }


    /* Initialize training and validation data */
    trainingdata->initialize(config->ntraining, config->nfeatures, config->nclasses, config->nbatch, MPI_COMM_WORLD);
    trainingdata->readData(config->datafolder, config->ftrain_ex, config->ftrain_labels);

    validationdata->initialize(config->nvalidation, config->nfeatures, config->nclasses, config->nvalidation, MPI_COMM_WORLD);  // full validation set!
    validationdata->readData(config->datafolder, config->fval_ex, config->fval_labels);


    /* Initialize XBraid */
    primaltrainapp = new myBraidApp(trainingdata, network, config, MPI_COMM_WORLD);
    adjointtrainapp = new myAdjointBraidApp(trainingdata, network, config, primaltrainapp->getCore(), MPI_COMM_WORLD);
    primalvalapp = new myBraidApp(validationdata, network, config, MPI_COMM_WORLD);


    /* Initialize the network  */
    primaltrainapp->GetGridDistribution(&ilower, &iupper);
    network->createNetworkBlock(ilower, iupper, config, MPI_COMM_WORLD);
    network->setInitialDesign(config);
    ndesign_local  = network->getnDesignLocal();
    ndesign_global = network->getnDesignGlobal();

    /* Print some network information */
    int startid = ilower;
    if (ilower == 0) startid = -1;
    printf("%d: Layer range: [%d, %d] / %d\n", myid, startid, iupper, config->nlayers);
    printf("%d: Design variables (local/global): %d/%d\n", myid, ndesign_local, ndesign_global);



    Optimizer* optimizer = new Optimizer();
    myFunction* function = new myFunction();
    


    /* Initialize hessian approximation */
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

    /* Initialize optimization parameters */
    ascentdir   = new MyReal[ndesign_local];
    stepsize    = config->getStepsize(0);
    gnorm       = 0.0;
    objective   = 0.0;
    rnorm       = 0.0;
    rnorm_adj   = 0.0;
    ls_param    = 1e-4;
    ls_iter     = 0;
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
        trainingdata->selectBatch(config->batch_type, MPI_COMM_WORLD);

        /* Solve state and adjoint equation */
        rnorm       = primaltrainapp->run();
        rnorm_adj = adjointtrainapp->run();

        /* Get output */
        objective   = primaltrainapp->getObjective();
        loss_train  = network->getLoss();
        accur_train = network->getAccuracy();


        /* --- Validation data: Get accuracy --- */

        if ( config->validationlevel > 0 )
        {
            primalvalapp->run();
            loss_val  = network->getLoss();
            accur_val = network->getAccuracy();
        }


        /* --- Optimization control and output ---*/

        /* Compute global gradient norm */
        gnorm = vecnorm_par(ndesign_local, network->getGradient(), MPI_COMM_WORLD);

        /* Communicate loss and accuracy. This is actually only needed for output. Remove it. */
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

                primaltrainapp->getCore()->SetPrintLevel(0);
                primaltrainapp->run();
                ls_objective = primaltrainapp->getObjective();
                primaltrainapp->getCore()->SetPrintLevel(config->braid_printlevel);

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

        primalvalapp->getCore()->SetPrintLevel(0);
        primalvalapp->run();
        loss_val  = network->getLoss();

        printf("Final validation accuracy:  %2.2f%%\n", accur_val);
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

    delete primaltrainapp;
    delete adjointtrainapp;
    delete primalvalapp;

    /* Delete optimization vars */
    delete hessian;
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

    delete function;
    delete optimizer;

    MPI_Finalize();
    return 0;
}
