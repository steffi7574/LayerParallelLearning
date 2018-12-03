#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "defs.hpp"
#pragma once

/**
 * Read data from file 
 */
void read_matrix(char*    filename, 
               MyReal** var, 
               int      dimx, 
               int      dimy);

/**
 * Read data from file 
 */
void read_vector(char*    filename, 
                 MyReal*  var, 
                 int      dimy);

/**
 * Write data to file
 */
void write_vector(char   *filename,
                  MyReal  *var, 
                  int      dimN);


/**
 * Gather a local vector of size localsendcount into global recvbuffer at root
 */
void MPI_GatherVector(MyReal*  sendbuffer,
                      int      localsendcount,
                      MyReal*  recvbuffer,
                      int      rootprocessID,
                      MPI_Comm comm); 
/**
 * Scatter parts of a global vector on root to local vectors on each processor (size localrecvsize)
 */
void MPI_ScatterVector(MyReal*  sendbuffer,
                      MyReal*  recvbuffer,
                      int      localrecvcount,
                      int      rootprocessID,
                      MPI_Comm comm);
