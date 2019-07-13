#include "util.hpp"

void read_matrix(char *filename, MyReal **var, int dimx, int dimy) {
  FILE *file;
  MyReal tmp;

  /* Open file */
  file = fopen(filename, "r");
  if (file == NULL) {
    printf("Can't open %s \n", filename);
    exit(1);
  }

  /* Read data */
  printf("Reading file %s\n", filename);
  for (int ix = 0; ix < dimx; ix++) {
    for (int iy = 0; iy < dimy; iy++) {
      fscanf(file, "%lf", &tmp);
      var[ix][iy] = tmp;
    }
  }

  fclose(file);
}

void read_vector(char *filename, MyReal *var, int dimx) {
  FILE *file;
  MyReal tmp;

  /* Open file */
  file = fopen(filename, "r");
  if (file == NULL) {
    printf("Can't open %s \n", filename);
    exit(1);
  }

  /* Read data */
  printf("Reading file %s\n", filename);
  for (int ix = 0; ix < dimx; ix++) {
    fscanf(file, "%lf", &tmp);
    var[ix] = tmp;
  }

  fclose(file);
}

void write_vector(char *filename, MyReal *var, int dimN) {
  FILE *file;
  int i;

  /* open file */
  file = fopen(filename, "w");
  if (file == NULL) {
    printf("Can't open %s \n", filename);
    exit(1);
  }

  /* Write data */
  printf("Writing file %s\n", filename);
  for (i = 0; i < dimN; i++) {
    fprintf(file, "%1.14e\n", var[i]);
  }

  /* close file */
  fclose(file);
}

void MPI_GatherVector(MyReal *sendbuffer, int localsendcount,
                      MyReal *recvbuffer, int rootprocessID, MPI_Comm comm) {
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int *recvcount = new int[comm_size];
  int *displs = new int[comm_size];

  /* Gather the local send counts and store in recvcount vector on root */
  MPI_Gather(&localsendcount, 1, MPI_INT, recvcount, 1, MPI_INT, rootprocessID,
             comm);

  /* Compute displacement vector */
  displs[0] = 0;
  for (int i = 1; i < comm_size; i++) {
    displs[i] = displs[i - 1] + recvcount[i - 1];
  }

  /* Gatherv the vector */
  MPI_Gatherv(sendbuffer, localsendcount, MPI_MyReal, recvbuffer, recvcount,
              displs, MPI_MyReal, rootprocessID, comm);

  /* Clean up */
  delete[] recvcount;
  delete[] displs;
}

void MPI_ScatterVector(MyReal *sendbuffer, MyReal *recvbuffer,
                       int localrecvcount, int rootprocessID, MPI_Comm comm) {
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int *sendcount = new int[comm_size];
  int *displs = new int[comm_size];

  /* Gather the local recveive counts and store in sendcount for root */
  MPI_Gather(&localrecvcount, 1, MPI_INT, sendcount, 1, MPI_INT, rootprocessID,
             comm);

  /* Compute displacement vector */
  displs[0] = 0;
  for (int i = 1; i < comm_size; i++) {
    displs[i] = displs[i - 1] + sendcount[i - 1];
  }

  /* Gatherv the vector */
  MPI_Scatterv(sendbuffer, sendcount, displs, MPI_MyReal, recvbuffer,
               localrecvcount, MPI_MyReal, rootprocessID, comm);

  /* Clean up */
  delete[] sendcount;
  delete[] displs;
}