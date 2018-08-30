#include <stdio.h>
#include <stdlib.h>
#pragma once

/**
 * Read data from file 
 */
void read_data(char*    filename, 
               double** var, 
               int      dimx, 
               int      dimy);

/**
 * Write data to file
 */
void write_data(char   *filename,
               double  *var, 
               int      dimN);
