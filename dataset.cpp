#include "dataset.hpp"

DataSet::DataSet()
{
   nelements = 0;
   nfeatures = 0;
   nlabels   = 0;
   nbatch    = 0;
   MPIsize   = 0;
   MPIrank   = 0;
   navail    = 0;

   examples = NULL;
   labels   = NULL;
   batchIDs = NULL;
   availIDs = NULL;
}

void DataSet::initialize(int      nElements, 
                         int      nFeatures, 
                         int      nLabels,
                         int      nBatch,
                         MPI_Comm comm)
{


   nelements = nElements;
   nfeatures = nFeatures;
   nlabels   = nLabels;
   nbatch    = nBatch;
   navail    = nelements;

   MPI_Comm_rank(comm, &MPIrank);
   MPI_Comm_size(comm, &MPIsize);

   /* Sanity check */
   if (nbatch > nelements) nbatch = nelements;

   /* Allocate feature vectors on first processor */
   if (MPIrank == 0)
   {
      examples = new MyReal*[nelements];
      for (int ielem = 0; ielem < nelements; ielem++)
      {
         examples[ielem] = new MyReal[nfeatures];
      }
   }
   /* Allocate label vectors on last processor */
   if (MPIrank == MPIsize - 1)
   {
      labels = new MyReal*[nelements];
      for (int ielem = 0; ielem < nelements; ielem++)
      {
         labels[ielem] = new MyReal[nlabels];
      }
   }

   /* Allocate and initialize availIDs and batchIDs on first and last processor */
   if (MPIrank == 0 || MPIrank == MPIsize - 1)
   {
      availIDs = new int[nelements];    // all elements 
      batchIDs = new int[nbatch];       

      /* Initialize available ID with identity */
      for (int idx = 0; idx < nelements; idx++)
      {
         availIDs[idx] = idx;
      }

      /* Initialize the batch with identity */
      for (int idx = 0; idx < nbatch; idx++)
      {
         batchIDs[idx] = idx;
      }
   }
}


DataSet::~DataSet()
{
   /* Deallocate feature vectors on first processor */
   if (examples != NULL)
   {
      for (int ielem = 0; ielem < nelements; ielem++)
      {
         delete [] examples[ielem];
      }
      delete [] examples;
   }

   /* Deallocate label vectors on last processor */
   if (labels != NULL)
   {
      for (int ielem = 0; ielem < nelements; ielem++)
      {
         delete [] labels[ielem];
      }
      delete [] labels;
   }

   if (availIDs != NULL) delete [] availIDs;
   if (batchIDs != NULL) delete [] batchIDs;
}


int DataSet::getnBatch() { return nbatch; }

MyReal* DataSet::getExample(int id) 
{ 
   if (examples == NULL) return NULL;

   return examples[batchIDs[id]]; 
}

MyReal* DataSet::getLabel(int id) 
{ 
   if (labels == NULL) return NULL;
   
   return labels[batchIDs[id]]; 
}

void DataSet::readData(char* datafolder,
                       char* examplefile, 
                       char* labelfile)
{
   char examplefilename[255], labelfilename[255];

   /* Set the file names  */
   sprintf(examplefilename,  "%s/%s", datafolder, examplefile);
   sprintf(labelfilename,    "%s/%s", datafolder, labelfile);

   /* Read feature vectors on first processor */
   if (MPIrank == 0) read_matrix(examplefilename, examples, nelements, nfeatures);

   /* Read label vectors on last processor) */
   if (MPIrank == MPIsize - 1) read_matrix(labelfilename, labels, nelements, nlabels);
}



void DataSet::selectBatch(int      batch_type, 
                          MPI_Comm comm)
{
   int irand, rand_range;
   int tmp;
   MPI_Request sendreq, recvreq;
   MPI_Status status;

   switch (batch_type)
   {
      case DETERMINISTIC:
         /* Do nothing, keep the batch fixed. */
         break;

      case STOCHASTIC:

         /* Randomly choose a batch on first processor, send to last processor */
         if (MPIrank == 0)
         {
            /* Fill the batchID vector with randomly generated integer */
            rand_range = navail - 1;
            for (int ibatch = 0; ibatch < nbatch; ibatch++)
            {
               /* Generate a new random index in [0,range] */
               irand = (int) ( ( ((double) rand()) /  (double) RAND_MAX ) * rand_range );

               /* Set the batchID */
               batchIDs[ibatch] = availIDs[irand];

               /* Remove the ID from available IDs (by swapping it with the last available id and reducing the range) */
               tmp = availIDs[irand];
               availIDs[irand] = availIDs[rand_range];
               availIDs[rand_range] = tmp;
               rand_range--;
            }

            /* Send to the last processor */
            int receiver = MPIsize - 1;
            MPI_Isend(batchIDs, nbatch, MPI_INT, receiver, 0, comm, &sendreq);
         }

         /* Receive the batch IDs on last processor */
         if (MPIrank == MPIsize - 1)
         {
            int source = 0;
            MPI_Irecv(batchIDs, nbatch, MPI_INT, source, 0, comm, &recvreq);
         }

         /* Wait to finish communication */
         if (MPIrank == 0)          MPI_Wait(&sendreq, &status);
         if (MPIrank == MPIsize-1)  MPI_Wait(&recvreq, &status);


         break; // break switch statement
   }
}


void DataSet::printBatch()
{
   if (batchIDs != NULL)  // only first and last processor
   {
      printf("%d:\n", MPIrank);
      for (int ibatch = 0; ibatch < nbatch; ibatch++)
      {
         printf("%d, %04d\n", ibatch, batchIDs[ibatch]);
      }
   }
}
