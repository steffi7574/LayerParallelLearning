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
// S. Guenther, L. Ruthotto, J.B. Schroder, E.C. Czr, and N.R. Gauger
//
// Download: https://arxiv.org/pdf/1812.04352.pdf
//
#include "dataset.hpp"

DataSet::DataSet(int nElements, int nFeatures, int nLabels, int nBatch) {
  nelements = nElements;
  nfeatures = nFeatures;
  nlabels = nLabels;
  nbatch = nBatch;

  navail = nElements;

  examples = NULL;
  labels = NULL;
  batchIDs = NULL;
  availIDs = NULL;

  /* Sanity check */
  if (nbatch > nelements) nbatch = nelements;

  lastlayer_proc = -1;

}

void DataSet::initBatch() {


  /* Allocate and initialize availIDs and batchIDs on first and last processor
   */
    availIDs = new int[nelements];  // all elements
    batchIDs = new int[nbatch];

    /* Initialize available ID with identity */
    for (int idx = 0; idx < nelements; idx++) {
      availIDs[idx] = idx;
    }

    /* Initialize the batch with identity */
    for (int idx = 0; idx < nbatch; idx++) {
      batchIDs[idx] = idx;
    }
}

DataSet::~DataSet() {
  /* Deallocate feature vectors */
  if (examples != NULL) {
    for (int ielem = 0; ielem < nelements; ielem++) {
      delete[] examples[ielem];
    }
    delete[] examples;
  }

  /* Deallocate label vectors */
  if (labels != NULL) {
    for (int ielem = 0; ielem < nelements; ielem++) {
      delete[] labels[ielem];
    }
    delete[] labels;
  }

  if (availIDs != NULL) delete[] availIDs;
  if (batchIDs != NULL) delete[] batchIDs;
}

int DataSet::getnBatch() { return nbatch; }



MyReal *DataSet::getExample(int id) {
  if (examples == NULL) return NULL;

  return examples[batchIDs[id]];
}

MyReal *DataSet::getLabel(int id) {
  if (labels == NULL) return NULL;

  return labels[batchIDs[id]];
}

MyReal **DataSet::getExamples(){
   return examples;
}

MyReal **DataSet::getLabels(){
   return labels;
}

void DataSet::loadExamples(const char *datafolder, const char *examplefile){

  char filename[255];

  /* Allocate the examples */
  examples = new MyReal *[nelements];
  for (int ielem = 0; ielem < nelements; ielem++) {
    examples[ielem] = new MyReal[nfeatures];
  }
 
  /* Read examples from file */
  sprintf(filename, "%s/%s", datafolder, examplefile);
  read_matrix(filename, examples, nelements, nfeatures);
}

void DataSet::loadLabels(const char *datafolder, const char *labelfile, int mpirank){

  char filename[255];

  /* Allocate label vector */
  labels = new MyReal *[nelements];
  for (int ielem = 0; ielem < nelements; ielem++) {
          labels[ielem] = new MyReal[nlabels];
  }

  /* Read label vector from file */
  sprintf(filename, "%s/%s", datafolder, labelfile);
  read_matrix(filename, labels, nelements, nlabels);

  /* Store the rank of the processor that stores the labels */
  lastlayer_proc = mpirank;
}

void DataSet::selectBatch(int batch_type, MPI_Comm comm) {
  int irand, rand_range;
  int tmp;
  MPI_Request sendreq, recvreq;
  MPI_Status status;
  int mpirank;
  MPI_Comm_rank(comm, &mpirank);

  switch (batch_type) {
    case DETERMINISTIC:
      /* Do nothing, keep the batch fixed. */
      break;

    case STOCHASTIC:

      /* Randomly choose a batch on first processor,  */
      if (mpirank == 0) {
        /* Fill the batchID vector with randomly generated integer */
        rand_range = navail - 1;
        for (int ibatch = 0; ibatch < nbatch; ibatch++) {
          /* Generate a new random index in [0,range] */
          irand = (int)((((double)rand()) / (double)RAND_MAX) * rand_range);

          /* Set the batchID */
          batchIDs[ibatch] = availIDs[irand];

          /* Remove the ID from available IDs (by swapping it with the last
           * available id and reducing the range) */
          tmp = availIDs[irand];
          availIDs[irand] = availIDs[rand_range];
          availIDs[rand_range] = tmp;
          rand_range--;
        }

        /* Send to the processor that stores the last layer */
        int receiver = lastlayer_proc;
        MPI_Isend(batchIDs, nbatch, MPI_INT, receiver, 0, comm, &sendreq);
      }

      /* Receive the batch IDs on last processor */
      if (MPIrank == lastlayer_proc) {
        int source = 0;
        MPI_Irecv(batchIDs, nbatch, MPI_INT, source, 0, comm, &recvreq);
      }

      /* Wait to finish communication */
      if (MPIrank == 0) MPI_Wait(&sendreq, &status);
      if (MPIrank == lastlayer_proc) MPI_Wait(&recvreq, &status);

      break;  // break switch statement
  }
}

void DataSet::printBatch() {
  if (batchIDs != NULL)  // only first and last processor
  {
    printf("%d:\n", MPIrank);
    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
      printf("%d, %04d\n", ibatch, batchIDs[ibatch]);
    }
  }
}
