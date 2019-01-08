#include "dataset.hpp"


DataSet::DataSet(int MPISize, 
                 int MPIRank,
                 int nElements, 
                 int nFeatures, 
                 int nLabels)
{
   MPIsize   = MPISize;
   MPIrank   = MPIRank;
   nelements = nElements;
   nfeatures = nFeatures;
   nlabels   = nLabels;

   examples = NULL;
   labels   = NULL;

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
}

DataSet::~DataSet()
{
   /* Deallocate feature vectors on first processor */
   if (MPIrank == 0)
   {
      for (int ielem = 0; ielem < nelements; ielem++)
      {
         delete [] examples[ielem];
      }
      delete [] examples;
   }

   /* Deallocate label vectors on last processor */
   if (MPIrank == MPIsize - 1)
   {
      for (int ielem = 0; ielem < nelements; ielem++)
      {
         delete [] labels[ielem];
      }
      delete [] labels;
   }

}


int DataSet::getnElements() { return nelements; }

MyReal* DataSet::getExample(int id) 
{ 
   if (examples == NULL) return NULL;
   
   return examples[id]; 
}

MyReal* DataSet::getLabel(int id) 
{ 
   if (labels == NULL) return NULL;
   
   return labels[id]; 
}

void DataSet::readData(char* examplefile, char* labelfile)
{
   /* Read feature vectors on first processor */
   if (MPIrank == 0) read_matrix(examplefile, examples, nelements, nfeatures);

   /* Read label vectors on last processor) */
   if (MPIrank == MPIsize - 1) read_matrix(labelfile, labels, nelements, nlabels);
}