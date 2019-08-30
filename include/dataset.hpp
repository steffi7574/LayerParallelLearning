#include <assert.h>
#include <mpi.h>
#include "config.hpp"
#include "defs.hpp"
#include "util.hpp"
#pragma once

class DataSet {
 protected:
  int nelements; /* Number of data elements */
  int nfeatures; /* Number of features per element */
  int nlabels;   /* Number of different labels (i.e. classes) per element */

  MyReal **examples; /* Array of Feature vectors (dim: nelements x nfeatures) */
  MyReal **labels;   /* Array of Label vectors (dim: nelements x nlabels) */

  int nbatch;    /* Size of the batch */
  int *batchIDs; /* Array of batch indicees */

  int MPIsize; /* Size of the global communicator */
  int MPIrank; /* Processors rank */

  int *availIDs; /* Auxilliary: holding available batchIDs when generating a
                    batch */
  int navail; /* Auxilliary: holding number of currently available batchIDs */

  int lastlayer_proc; /* Stores the MPI_rank of the processor that holds the last layer */

 public:
  /* Default constructor */
  DataSet(int nElements, int nFeatures, int nLabels, int nBatch);
  DataSet();

  /* Destructor */
  ~DataSet();

  /* Allocates the batchIDs and initializes with identity */
  void initBatch();

  /* Allocates vector of examples and read examples from the file*/
  void loadExamples(const char *datafolder, const char *examplefile);

  /* Allocates vector of labels and read labels from the file.
   * Also, stores the rank of the processor who owns the labels. 
   */
  void loadLabels(const char *datafolder, const char *labelfile, int lastprocID);

  /* Return the batch size*/
  int getnBatch();

  /* Return the feature vector of a certain batchID. If not stored on this
   * processor, return NULL */
  MyReal *getExample(int id);

  /* Return a pointer to the vector of examples */
  MyReal **getExamples();

  /* Return a pointer to the vector of labels */
  MyReal **getLabels();

  /* Return the label vector of a certain batchID. If not stored on this
   * processor, return NULL */
  MyReal *getLabel(int id);

  /* Select the current batch from all available IDs, either deterministic or
   * stochastic */
  void selectBatch(int batch_type, MPI_Comm comm);

  /* print current batch to screen */
  void printBatch();
};