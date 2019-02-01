#include "dataset.hpp"

DataSet::DataSet()
{
   nelements = 0;
   nfeatures = 0;
   nlabels   = 0;
   nbatch    = 0;
   MPIsize   = 0;
   MPIrank   = 0;

   batch_type = -1;
   batchindex = -1;
   epochiter  = -1;

   examples = NULL;
   labels   = NULL;
   batchIDs = NULL;
   allIDs   = NULL;
   // availIDs = NULL;
}

void DataSet::initialize(int      nElements, 
                         int      nFeatures, 
                         int      nLabels,
                         int      nBatch,
                         int      batchType,
                         MPI_Comm comm)
{
   nelements  = nElements;
   nfeatures  = nFeatures;
   nlabels    = nLabels;
   nbatch     = nBatch;
   batch_type = batchType;

   if (nbatch >= nelements)
   {
      /* Full batch -> deterministic batch selection */
      nbatch     = nelements;
      batch_type = DETERMINISTIC;
   }

   /* Sanity check */
   if (nelements % nbatch != 0) 
   {
      printf("\n\n ERROR: Number of data set elements must be an integer multiple of the batch size!\n\n ");
      exit(1);
   }

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

   /* Allocate and initialize allIDs with identity on first and last processor */
   if (MPIrank == 0 || MPIrank == MPIsize - 1)
   {
      allIDs   = new int[nelements];    // all elements 

      /* Initialize */
      for (int idx = 0; idx < nelements; idx++)
      {
         allIDs[idx] = idx;
      }
   }

   /* First batch points to beginning of allIDs */
   batchIDs = allIDs;
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

   if (allIDs != NULL) delete [] allIDs;
}


int DataSet::getnBatch() { return nbatch; }


int DataSet::getEpochIter() { return epochiter; }
      
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



void DataSet::selectBatch(int      iter,
                          MPI_Comm comm)
{
   /* in first iteration, shuffle all IDs */
   if (iter == 0)
   {
      shuffle(comm);
   }

   switch (batch_type)
   {
      case DETERMINISTIC:
         
         /* Do nothing. Keep batch fixed. */

         /* Set epoch for output */
         epochiter = 0;
         if (nbatch >= nelements) epochiter = iter;
         
         break;

      case STOCHASTIC:

         /* Set batchIDs pointer to the next batch in allIDs */
         batchindex = iter % (nelements / nbatch);
         batchIDs   = &(allIDs[batchindex * nbatch]); 

         /* if a full epoch has finished (or first iter), shuffle allIDs  */
         if ( batchindex == 0)
         {
            shuffle(comm);
            epochiter++;
         }

         break; // of switch statement
   }
}

void DataSet::printBatch()
{
   if (MPIrank == 0 )
   {
      printf("Batch "); 
      print_int_vector(batchIDs, nbatch);
   }
}


void DataSet::shuffle(MPI_Comm Comm)
{
   MPI_Request sendreq, recvreq;
   MPI_Status status;

   // /* Shuffle allIDs on first processor and send to last */
   // if (MPIrank == 0)
   // {
   //    printf("Shuffle \n");
   //    shuffle_int_vector(allIDs, nelements);
   //    // print_int_vector(allIDs, nelements);

   //    int receiver = MPIsize - 1;
   //    MPI_Isend(allIDs, nelements, MPI_INT, receiver, 0, Comm, &sendreq);
   // }
   // if (MPIrank == MPIsize - 1)
   // {
   //    int source = 0;
   //    MPI_Irecv(allIDs, nelements, MPI_INT, source, 0, Comm, &recvreq);
   // }

   // if (MPIrank == 0        ) MPI_Wait(&sendreq, &status);
   // if (MPIrank == MPIsize-1) MPI_Wait(&recvreq, &status);


   /* Shuffle allIDs on first processor and send to last */
   int N = 243;
   int* someint = new int[N];
   printf("%d: nelements %d\n", MPIrank, nelements);

   if (MPIrank == 0)
   {
      // printf("Shuffle \n");
      // shuffle_int_vector(allIDs, nelements);
      // print_int_vector(allIDs, nelements);

      printf("before Shuffle %d \n", MPIsize);
      int receiver = MPIsize - 1;
      MPI_Send(someint, N, MPI_INT, receiver, 0, MPI_COMM_WORLD);
   }
   printf("Mid Shuffle \n");
   if (MPIrank == MPIsize - 1)
   {
      int source = 0;
      MPI_Recv(someint, N, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
   }

   printf("End Shuffle \n");
}