#include "util.hpp"
#include "defs.hpp"
#pragma once

class DataSet {

   protected:
      int MPIsize;           /* Size of the global communicator */
      int MPIrank;           /* Processors rank */

      int nelements;         /* Number of data elements */
      int nfeatures;         /* Number of features per element */
      int nlabels;           /* Number of different labels (i.e. classes) per element */
      
      MyReal **examples;    /* Array of Feature vectors (dim: nelements x nfeatures) */
      MyReal **labels;      /* Array of Label vectors (dim: nelements x nlabels) */

   public: 

      /* Constructor */
      DataSet(int MPISize, 
              int MPIRank,
              int nElements, 
              int nFeatures, 
              int nLabels);

      /* Destructor */
      ~DataSet();

      /* Return the number of elements in the data set */
      int getnElements();


      /* Return the feature vector of a certain id, if stored on this processor (else NULL) */
      MyReal* getExample(int id);

      /* Return the label vector of a certain id, if stored on this processor (else NULL)*/
      MyReal* getLabel(int id);

      /* Read data from file */
      void readData(char* examplefile,
                    char* labelfile);
};