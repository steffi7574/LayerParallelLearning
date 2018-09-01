#include "util.hpp"

void read_matrix(char    *filename, 
               double **var, 
               int      dimx, 
               int      dimy)
{
   FILE   *file;
   double  tmp;

   /* Open file */
   file = fopen(filename, "r");
   if (file == NULL)
   {
      printf("Can't open %s \n", filename);
      exit(1);
   }

   /* Read data */
   printf("Reading file %s\n", filename);
   for (int ix = 0; ix < dimx; ix++)
   {
       for (int iy = 0; iy < dimy; iy++)
       {
            fscanf(file, "%lf", &tmp);
            var[ix][iy] = tmp;
       }
   }

   fclose(file);
}

void read_vector(char *filename, 
                 double *var, 
                 int      dimx)
{
   FILE   *file;
   double  tmp;

   /* Open file */
   file = fopen(filename, "r");
   if (file == NULL)
   {
      printf("Can't open %s \n", filename);
      exit(1);
   }

   /* Read data */
   printf("Reading file %s\n", filename);
   for (int ix = 0; ix < dimx; ix++)
   {
            fscanf(file, "%lf", &tmp);
            var[ix] = tmp;
   }

   fclose(file);
}


void write_vector(char   *filename,
                  double * var, 
                  int      dimN)
{
   FILE *file;
   int i;

   /* open file */
   file = fopen(filename, "w");
   if (file == NULL)
   {
      printf("Can't open %s \n", filename);
      exit(1);
   }

   /* Write data */
   printf("Writing file %s\n", filename);
   for ( i = 0; i < dimN; i++)
   {
      fprintf(file, "%1.14e\n", var[i]);
   }

   /* close file */
   fclose(file);

}            
