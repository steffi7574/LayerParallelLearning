#include <mpi.h>
#pragma once

/*
 * Switch between single (float) and double precision by un-/commenting the
 * corresponding lines.
 */

// typedef float MyReal;
// #define MPI_MyReal MPI_FLOAT
typedef double MyReal;
#define MPI_MyReal MPI_DOUBLE