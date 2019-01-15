#include <assert.h>
#include "util.hpp"
#include "defs.hpp"
#include "config.hpp"
#include <mpi.h>
#pragma once

class DataSet {


   protected:

      int nelements;         /* Number of data elements */
      int nfeatures;         /* Number of features per element */
      int nlabels;           /* Number of different labels (i.e. classes) per element */
      
      MyReal **examples;    /* Array of Feature vectors (dim: nelements x nfeatures) */
      MyReal **labels;      /* Array of Label vectors (dim: nelements x nlabels) */

      int  nbatch;          /* Size of the batch */
      int *batchIDs;        /* Array of batch indicees */

      int MPIsize;           /* Size of the global communicator */
      int MPIrank;           /* Processors rank */

      int* availIDs;          /* Auxilliary: holding available batchIDs when generating a batch */
      int  navail;            /* Auxilliary: holding number of currently available batchIDs */

   public: 

      /* Default constructor */
      DataSet();

      /* Destructor */
      ~DataSet();

      void initialize(int      nElements, 
                      int      nFeatures, 
                      int      nLabels,
                      int      nBatch,
                      MPI_Comm Comm);


      /* Return the batch size*/
      int getnBatch();

      /* Return the feature vector of a certain batchID. If not stored on this processor, return NULL */
      MyReal* getExample(int id);

      /* Return the label vector of a certain batchID. If not stored on this processor, return NULL */
      MyReal* getLabel(int id);

      /* Read data from file */
      void readData(char* datafolder,
                    char* examplefile,
                    char* labelfile);

      /* Select the current batch from all available IDs, either deterministic or stochastic */
      void selectBatch(int      batch_type,
                       MPI_Comm comm);


      /* print current batch to screen */
      void printBatch();
};