#include "linalg.hpp"

MyReal vecdot_par(int dimN, MyReal *x, MyReal *y, MPI_Comm comm) {
  MyReal localdot, globaldot;

  localdot = vecdot(dimN, x, y);
  MPI_Allreduce(&localdot, &globaldot, 1, MPI_MyReal, MPI_SUM, comm);

  return globaldot;
}

MyReal vecdot(int dimN, MyReal *x, MyReal *y) {
  MyReal dotprod = 0.0;
  for (int i = 0; i < dimN; i++) {
    dotprod += x[i] * y[i];
  }
  return dotprod;
}

MyReal vecmax(int dimN, MyReal *x) {
  MyReal max = -1e+12;

  for (int i = 0; i < dimN; i++) {
    if (x[i] > max) {
      max = x[i];
    }
  }
  return max;
}

int argvecmax(int dimN, MyReal *x) {
  MyReal max = -1e+12;
  int i_max;
  for (int i = 0; i < dimN; i++) {
    if (x[i] > max) {
      max = x[i];
      i_max = i;
    }
  }
  return i_max;
}

MyReal vecnormsq(int dimN, MyReal *x) {
  MyReal normsq = 0.0;
  for (int i = 0; i < dimN; i++) {
    normsq += pow(x[i], 2);
  }
  return normsq;
}

MyReal vecnorm_par(int dimN, MyReal *x, MPI_Comm comm) {
  MyReal localnorm, globalnorm;

  localnorm = vecnormsq(dimN, x);
  MPI_Allreduce(&localnorm, &globalnorm, 1, MPI_MyReal, MPI_SUM, comm);
  globalnorm = sqrt(globalnorm);

  return globalnorm;
}

int vec_copy(int N, MyReal *u, MyReal *u_copy) {
  for (int i = 0; i < N; i++) {
    u_copy[i] = u[i];
  }

  return 0;
}

void vecvecT(int N, MyReal *x, MyReal *y, MyReal *XYT) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      XYT[i * N + j] = x[i] * y[j];
    }
  }
}

void matvec(int dimN, MyReal *H, MyReal *x, MyReal *Hx) {
  MyReal sum_j;

  for (int i = 0; i < dimN; i++) {
    sum_j = 0.0;
    for (int j = 0; j < dimN; j++) {
      sum_j += H[i * dimN + j] * x[j];
    }
    Hx[i] = sum_j;
  }
}
