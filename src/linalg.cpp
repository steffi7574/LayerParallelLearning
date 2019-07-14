#include "linalg.hpp"

double vecdot_par(int dimN, double *x, double *y, MPI_Comm comm) {
  double localdot, globaldot;

  localdot = vecdot(dimN, x, y);
  MPI_Allreduce(&localdot, &globaldot, 1, MPI_DOUBLE, MPI_SUM, comm);

  return globaldot;
}

double vecdot(int dimN, double *x, double *y) {
  double dotprod = 0.0;
  for (int i = 0; i < dimN; i++) {
    dotprod += x[i] * y[i];
  }
  return dotprod;
}

double vecmax(int dimN, double *x) {
  double max = -1e+12;

  for (int i = 0; i < dimN; i++) {
    if (x[i] > max) {
      max = x[i];
    }
  }
  return max;
}

int argvecmax(int dimN, double *x) {
  double max = -1e+12;
  int i_max;
  for (int i = 0; i < dimN; i++) {
    if (x[i] > max) {
      max = x[i];
      i_max = i;
    }
  }
  return i_max;
}

double vecnormsq(int dimN, double *x) {
  double normsq = 0.0;
  for (int i = 0; i < dimN; i++) {
    normsq += pow(x[i], 2);
  }
  return normsq;
}

double vecnorm_par(int dimN, double *x, MPI_Comm comm) {
  double localnorm, globalnorm;

  localnorm = vecnormsq(dimN, x);
  MPI_Allreduce(&localnorm, &globalnorm, 1, MPI_DOUBLE, MPI_SUM, comm);
  globalnorm = sqrt(globalnorm);

  return globalnorm;
}

int vec_copy(int N, double *u, double *u_copy) {
  for (int i = 0; i < N; i++) {
    u_copy[i] = u[i];
  }

  return 0;
}

void vecvecT(int N, double *x, double *y, double *XYT) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      XYT[i * N + j] = x[i] * y[j];
    }
  }
}

void matvec(int dimN, double *H, double *x, double *Hx) {
  double sum_j;

  for (int i = 0; i < dimN; i++) {
    sum_j = 0.0;
    for (int j = 0; j < dimN; j++) {
      sum_j += H[i * dimN + j] * x[j];
    }
    Hx[i] = sum_j;
  }
}
