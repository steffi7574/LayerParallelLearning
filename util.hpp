#include <stdio.h>
#include <stdlib.h>
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
