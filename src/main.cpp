// Copyright
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Underlying paper:
//
// Layer-Parallel Training of Deep Residual Neural Networks
// S. Guenther, L. Ruthotto, J.B. Schroder, E.C. Cyr, and N.R. Gauger
//
// Download: https://arxiv.org/pdf/1812.04352.pdf
//
#include <mpi.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "braid_wrapper.hpp"
#include "config.hpp"
#include "dataset.hpp"
#include "defs.hpp"
#include "hessianApprox.hpp"
#include "layer.hpp"
#include "network.hpp"
#include "util.hpp"

#define MASTER_NODE 0

int main(int argc, char *argv[]) {
  /* --- Data --- */
  Config *config;          /**< Storing configurations */
  DataSet *trainingdata;   /**< Training dataset */
  DataSet *validationdata; /**< Validation dataset */

  /* --- Nested Iteration --- */
  std::vector<Network*>  vnetworks;     /**< Vector of networks */ 
  int current_nhiddenlayers;            /**< nhiddenlayers to be used on the current NI level */
  int startlayerID, endlayerID; /**< Index of first and last layer stored on this processor */

  /* --- Other Network Values --- */
  MyReal accur_train = 0.0; /**< Accuracy on training data */
  MyReal accur_val = 0.0;   /**< Accuracy on validation data */
  MyReal loss_train = 0.0;  /**< Loss function on training data */
  MyReal loss_val = 0.0;    /**< Loss function on validation data */
  MyReal losstrain_out = 0.0;
  MyReal lossval_out = 0.0;
  MyReal accurtrain_out = 0.0;
  MyReal accurval_out = 0.0;

  /* --- XBraid --- */
  myBraidApp *primaltrainapp;         /**< Braid App for training data */
  myAdjointBraidApp *adjointtrainapp; /**< Adjoint Braid for training data */
  myBraidApp *primalvalapp;           /**< Braid App for validation data */

  /* --- Optimization --- */
  int ndesign_local;  /**< Number of local design variables on this processor */
  int ndesign_global; /**< Number of global design variables (sum of local)*/
  MyReal *ascentdir = 0; /**< Direction for design updates */
  MyReal objective;      /**< Optimization objective */
  MyReal wolfe;          /**< Holding the wolfe condition value */
  MyReal rnorm;          /**< Space-time Norm of the state variables */
  MyReal rnorm_adj;      /**< Space-time norm of the adjoint variables */
  MyReal gnorm;          /**< Norm of the gradient */
  MyReal ls_param;       /**< Parameter in wolfe condition test */
  MyReal stepsize;       /**< Stepsize used for design update */
  char optimfilename[255];
  FILE *optimfile = 0;
  MyReal ls_stepsize;
  MyReal ls_objective, test_obj;
  int ls_iter;
  
  /* --- Time measurements --- */
  struct rusage r_usage;
  MyReal StartTime, StopTime, myMB, globalMB;
  MyReal UsedTime = 0.0;

  /* Initialize MPI */
  int myid;
  int size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  /* Read config file */
  config = new Config();
  if (argc != 2) {
    if (myid == MASTER_NODE) {
      printf("\n");
      printf("USAGE: ./main </path/to/configfile> \n");
    }
    MPI_Finalize();
    return 0;
  }
  int err = config->readFromFile(argv[1]);
  if (err) {
    printf("Error while reading config file!\n");
    MPI_Finalize();
    return 0;
  }

  /* Initialize vector of networks for nested iteration */
  for(int i = 0; i < config->NI_levels; i++)
  {
     vnetworks.push_back( new Network(MPI_COMM_WORLD) );
  }


  /* Initialize the number of layers for the first level of NI */
  current_nhiddenlayers = config->nlayers - 2;

  /* Initialize training and validation data */
  trainingdata = new DataSet(config->ntraining, config->nfeatures, config->nclasses, config->nbatch);
  validationdata = new DataSet(config->nvalidation, config->nfeatures, config->nclasses, config->nbatch);// full validation set!

  /* Initialize and open optimization data file */
  if (myid == MASTER_NODE) {
     sprintf(optimfilename, "%s.dat", "optim");
     optimfile = fopen(optimfilename, "w");
     config->writeToFile(optimfile);
  }

  /* Measure wall time */
  StartTime = MPI_Wtime();
  StopTime = 0.0;
  UsedTime = 0.0;

  /** Nested Iteration Loop 
   *
   *  Each nested iteration refines the network by config->NI_rfactor, 
   *  by adding more layers.  The new refined network is initialized by 
   *  interpolating the previous design variable (i.e., network weights 
   *  and biases)to the new refined network
   *
   * */
  for(int NI_iter = 0; NI_iter < config->NI_levels; NI_iter++)
  {
     if (myid == MASTER_NODE) {
        printf("\n------------------------ Begin Nested Iteration %d------------------------\n\n", NI_iter);
     }

     /* Compute number of hidden layers for the next level of NI */
     if(NI_iter > 0) {
         current_nhiddenlayers *= config->NI_rfactor; 
     } 

     /* Initialize XBraid */
     primaltrainapp =
         new myBraidApp(trainingdata, vnetworks[NI_iter], config, MPI_COMM_WORLD, current_nhiddenlayers);
     adjointtrainapp = new myAdjointBraidApp(
         trainingdata, vnetworks[NI_iter], config, primaltrainapp->getCore(), MPI_COMM_WORLD, current_nhiddenlayers);
     primalvalapp =
         new myBraidApp(validationdata, vnetworks[NI_iter], config, MPI_COMM_WORLD, current_nhiddenlayers);
     primaltrainapp->GetGridDistribution(&startlayerID, &endlayerID);
     if (startlayerID == 0) startlayerID = startlayerID - 1; // -1 is index of the opening layer

     /* Read training and validation data */
     if (startlayerID == -1) // processor that stores opening layer
     {
        if (trainingdata->getExamples() == NULL){// if it hasn't been allocated before on this proc
          printf("%d: Reading examples...\n", myid);
          trainingdata->loadExamples(config->datafolder, config->ftrain_ex);
          validationdata->loadExamples(config->datafolder, config->fval_ex);
        }
     }
     if (endlayerID == current_nhiddenlayers && startlayerID <= endlayerID) // processor that stores classification layer
     {
        if (trainingdata->getLabels() == NULL){  // if it hasn't been allocated before on this proc
          printf("%d: Reading labels...\n", myid);
          trainingdata->loadLabels(config->datafolder, config->ftrain_labels, myid);
          validationdata->loadLabels(config->datafolder, config->fval_labels, myid);
        }
     }

     /* Allocate and initialize batch on processors that store opening or classification layer */
     if (startlayerID == -1 || (endlayerID == current_nhiddenlayers && startlayerID <= endlayerID) ) {
       trainingdata->initBatch();
       validationdata->initBatch();
     }

     /* Create the network */
     vnetworks[NI_iter]->createLayerBlock(startlayerID, endlayerID, config, current_nhiddenlayers);
     ndesign_local = vnetworks[NI_iter]->getnDesignLocal();
     ndesign_global = vnetworks[NI_iter]->getnDesignGlobal();
     printf("%d: Layer range: [%d, %d] / %d\n", myid, startlayerID, endlayerID, current_nhiddenlayers+2);
     printf("%d: Design variables (local/global): %d/%d\n", myid, ndesign_local, ndesign_global);

     /* Initialize network weights */
     if (NI_iter == 0){
        /* Init coarsest grid with scaled random vars, or from file, if set */
        vnetworks[NI_iter]->setDesignRandom(config->weights_open_init, config->weights_init,  config->weights_class_init);
        vnetworks[NI_iter]->setDesignFromFile(config->datafolder, config->weightsopenfile, NULL,  config->weightsclassificationfile);
     } else {      
        /* Interpolate from coarser to finer grid */
        vnetworks[NI_iter]->interpolateDesign(config->NI_rfactor, vnetworks[NI_iter-1], config->NI_interp_type);
     }

    // char designfilename[255];
    // sprintf(designfilename, "design_NI%d.dat", NI_iter);
    // write_vector(designfilename, vnetworks[NI_iter]->getDesign(), ndesign_global);

     // TODO:  If NI_iter > 0:  Do we want to deallocate the previous network?  Don't do for now, unles we run into memory issues. 


     

     /* Initialize Hessian approximation */
     HessianApprox *hessian = NULL;
     switch (config->hessianapprox_type) {
       case BFGS_SERIAL:
         hessian = new BFGS(MPI_COMM_WORLD, ndesign_local);
         break;
       case LBFGS:
         hessian = new L_BFGS(MPI_COMM_WORLD, ndesign_local, config->lbfgs_stages);
         break;
       case IDENTITY:
         hessian = new Identity(MPI_COMM_WORLD, ndesign_local);
         break;
       default:
         printf("Error: unexpected hessianapprox_type returned");
         return 0;
     }

     /* Initialize optimization parameters */
     ascentdir = new MyReal[ndesign_local];
     stepsize = config->getStepsize(0);
     gnorm = 0.0;
     objective = 0.0;
     rnorm = 0.0;
     rnorm_adj = 0.0;
     ls_param = 1e-4;
     ls_iter = 0;
     ls_stepsize = stepsize;

     /* Open and prepare optimization output file*/
     if (myid == MASTER_NODE) {
       fprintf(optimfile, "\n Begin Nested Iteration %d\n", NI_iter);
       fprintf(optimfile,
               "#    || r ||          || r_adj ||      Objective             Loss "
               "                 || grad ||            Stepsize  ls_iter   "
               "Accur_train  Accur_val   Time(sec)\n");
     }


     /** Main optimization iteration
      *
      * The following loop represents the paper's Algorithm (2)
      *
      */
     for (int iter = 0; iter < config->maxoptimiter[NI_iter]; iter++) {
       /* Set up the current batch. Only those processors that store opening or classification layer */
       if (startlayerID == -1 || endlayerID == current_nhiddenlayers) 
          trainingdata->selectBatch(config->batch_type, MPI_COMM_WORLD);

       /** Solve state and adjoint equations (2.15) and (2.17)
        *
        *  Algorithm (2): Step 1 and 2
        */
       rnorm = primaltrainapp->run();
       rnorm_adj = adjointtrainapp->run();

       /* Get output */
       objective = primaltrainapp->getObjective();
       loss_train = vnetworks[NI_iter]->getLoss();
       accur_train = vnetworks[NI_iter]->getAccuracy();

       /* --- Validation data: Get accuracy --- */
       if (config->validationlevel > 0) {
         primalvalapp->run();
         loss_val = vnetworks[NI_iter]->getLoss();
         accur_val = vnetworks[NI_iter]->getAccuracy();
       }

       /* --- Optimization control and output ---*/

       /** Compute global gradient norm
        *
        *  Algorithm (2): Step 3
        */
       gnorm = vecnorm_par(ndesign_local, vnetworks[NI_iter]->getGradient(), MPI_COMM_WORLD);

       /* Communicate loss and accuracy. Some of this information is used for early halting in NI */
       MPI_Allreduce(&loss_train, &losstrain_out, 1, MPI_MyReal, MPI_SUM,
                     MPI_COMM_WORLD);
       MPI_Allreduce(&loss_val, &lossval_out, 1, MPI_MyReal, MPI_SUM,
                     MPI_COMM_WORLD);
       MPI_Allreduce(&accur_train, &accurtrain_out, 1, MPI_MyReal, MPI_SUM,
                     MPI_COMM_WORLD);
       MPI_Allreduce(&accur_val, &accurval_out, 1, MPI_MyReal, MPI_SUM,
                     MPI_COMM_WORLD);

       /* Output */
       StopTime = MPI_Wtime();
       UsedTime = StopTime - StartTime;
       if (myid == MASTER_NODE) {
         printf(
             "\niter \t|| r ||\t\t|| r_adj ||\tObjective\t\tLoss\t\t\t|| grad "
             "||\t\tStepsize\tls_iter\tAccur_train\tAccur_val\tTime(sec)\n");
         printf(
             "%03d\t%1.8e\t%1.8e\t%1.14e\t%1.14e\t%1.14e\t%5f\t%2d\t%2.2f%%\t\t%2."
             "2f%%\t\t%.1f\n\n",
             iter, rnorm, rnorm_adj, objective, losstrain_out, gnorm, stepsize,
             ls_iter, accurtrain_out, accurval_out, UsedTime);
         fprintf(optimfile,
                 "%03d  %1.8e  %1.8e  %1.14e  %1.14e  %1.14e  %5f  %2d        "
                 "%2.2f%%      %2.2f%%     %.1f\n",
                 iter, rnorm, rnorm_adj, objective, losstrain_out, gnorm, stepsize,
                 ls_iter, accurtrain_out, accurval_out, UsedTime);
         fflush(optimfile);
       }

       /** Check optimization convergence
        *
        *  Algorithm (2): Step 6
        */
       if (gnorm < config->gtol) {
         if (myid == MASTER_NODE) {
           printf("Optimization has converged. \n");
           printf("Be happy and go home!       \n");
         }
         break;
       }
       if (iter == config->maxoptimiter[NI_iter] - 1) {
         if (myid == MASTER_NODE) {
           printf("\nMax. optimization iterations reached.\n");
         }
         break;
       }
       if ( (accurtrain_out > config->NI_tols[NI_iter]) && (NI_iter != (config->NI_levels-1) ) ){
         if (myid == MASTER_NODE) {
           printf("\nNI tolerance reached.\n"); 
         }
         break;
       }

       /* If optimization didn't converge, continue */

       /* --- Design update --- */

       /** Compute search direction
        *
        *  Algorithm (2): Step 4
        */
       hessian->updateMemory(iter, vnetworks[NI_iter]->getDesign(), vnetworks[NI_iter]->getGradient());
       hessian->computeAscentDir(iter, vnetworks[NI_iter]->getGradient(), ascentdir);
       stepsize = config->getStepsize(iter);

       /** Update the design/network control parameter in negative ascent direction
        *  and perform backtracking linesearch.
        *
        *  Algorithm (2): Step 5
        */
       vec_axpy(ndesign_local, -1.0*stepsize, ascentdir, vnetworks[NI_iter]->getDesign());
       vnetworks[NI_iter]->MPI_CommunicateNeighbours();

       if (config->stepsize_type == BACKTRACKINGLS) {
         /* Compute wolfe condition */
         wolfe = vecdot_par(ndesign_local, vnetworks[NI_iter]->getGradient(), ascentdir,
                            MPI_COMM_WORLD);

         /* Start linesearch iterations */
         ls_stepsize = config->getStepsize(iter);
         stepsize = ls_stepsize;
         for (ls_iter = 0; ls_iter < config->ls_maxiter; ls_iter++) {
           primaltrainapp->getCore()->SetPrintLevel(0);
           primaltrainapp->run();
           ls_objective = primaltrainapp->getObjective();
           primaltrainapp->getCore()->SetPrintLevel(config->braid_printlevel);

           test_obj = objective - ls_param * ls_stepsize * wolfe;
           if (myid == MASTER_NODE)
             printf("ls_iter = %d:\tls_objective = %1.14e\ttest_obj = %1.14e\n",
                    ls_iter, ls_objective, test_obj);
           /* Test the wolfe condition */
           if (ls_objective <= test_obj) {
             /* Success, use this new design */
             break;
           } else {
             /* Test for line-search failure */
             if (ls_iter == config->ls_maxiter - 1) {
               if (myid == MASTER_NODE)
                 printf("\n\n   WARNING: LINESEARCH FAILED! \n\n");
               break;
             }

             /* Go back part of the step */
             vec_axpy(ndesign_local, (1.0 - config->ls_factor) * stepsize, ascentdir, vnetworks[NI_iter]->getDesign());
             vnetworks[NI_iter]->MPI_CommunicateNeighbours();

             /* Decrease the stepsize */
             ls_stepsize = ls_stepsize * config->ls_factor;
             stepsize = ls_stepsize;
           }
         }
       }
     }


     /* --- Run final validation and write prediction file --- */
     if (config->validationlevel > -1) {
       if (myid == MASTER_NODE) printf("\n --- Run final validation ---\n");

       primalvalapp->getCore()->SetPrintLevel(0);
       primalvalapp->run();
       loss_val = vnetworks[NI_iter]->getLoss();
       accur_val = vnetworks[NI_iter]->getAccuracy();
       MPI_Allreduce(&accur_val, &accurval_out, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);

       if (myid == MASTER_NODE) printf("Final validation accuracy:  %2.2f%%\n", accurval_out);
     }

     /* Clean up XBraid */
     delete primaltrainapp;
     delete adjointtrainapp;
     delete primalvalapp;
     
     /* Delete optimization vars */
     delete hessian;
     delete[] ascentdir;

  } // End NI Loop
    
  // write_vector("design.dat", design, ndesign);

  /* Print some statistics */
  StopTime = MPI_Wtime();
  UsedTime = StopTime - StartTime;
  getrusage(RUSAGE_SELF, &r_usage);
  myMB = (MyReal)r_usage.ru_maxrss / 1024.0;
  MPI_Allreduce(&myMB, &globalMB, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);

  // printf("%d; Memory Usage: %.2f MB\n",myid, myMB);
  if (myid == MASTER_NODE) {
    printf("\n");
    printf(" Used Time:        %.2f seconds\n", UsedTime);
    printf(" Global Memory:    %.2f MB\n", globalMB);
    printf(" Processors used:  %d\n", size);
    printf("\n");
  }

  // Deallocate the vector of networks
  for(long unsigned int i = 0; i < vnetworks.size(); i++)
  {
      delete vnetworks[i];
  }
  vnetworks.clear();

  /* Delete training and validation examples  */
  delete trainingdata;
  delete validationdata;

  /* Close optim file */
  if (myid == MASTER_NODE) {
    fclose(optimfile);
    printf("Optimfile: %s\n", optimfilename);
  }

  delete config;

  MPI_Finalize();
  return 0;
}
