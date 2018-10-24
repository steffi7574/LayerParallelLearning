#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#pragma once

/**
 * Read data from file 
 */
void read_matrix(char*    filename, 
               double** var, 
               int      dimx, 
               int      dimy);

/**
 * Read data from file 
 */
void read_vector(char*    filename, 
                 double*  var, 
                 int      dimy);

/**
 * Write data to file
 */
void write_vector(char   *filename,
                  double  *var, 
                  int      dimN);


/**
 * Gather a local vector of size localsendcount into global recvbuffer at root
 */
void MPI_GatherVector(double*  sendbuffer,
                      int      localsendcount,
                      double*  recvbuffer,
                      int      rootprocessID,
                      MPI_Comm comm);