#include "util.hpp"

void read_data(char    *filename, 
               double **var, 
               int      dimx, 
               int      dimy)
{
   FILE   *file;
   double  tmp;

   /* Read data */
   file = fopen(filename, "r");
   if (file == NULL)
   {
      printf("Can't open %s \n", filename);
      exit(1);
   }
   printf("Reading file %s\n", filename);
   for (int ix = 0; ix < dimx; ix++)
   {
       for (int iy = 0; iy < dimy; iy++)
       {
            fscanf(file, "%lf", &tmp);
            var[ix][iy] = tmp;
            // printf("%1.14e ", var[ix][iy]);
       }
      //  printf("\n");
   }

   /* close file */
   fclose(file);
}


void write_data(char   *filename,
               double** var, 
               int      dimx, 
               int      dimy)
{
   printf("To be implemented ... \n");
}            
