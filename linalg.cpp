#include "linalg.hpp"

void matvec(int dimN,
            double* H, 
            double* x,
            double* Hx)
{
    double sum_j;

    for (int i=0; i<dimN; i++)
    {
       sum_j = 0.0;
       for (int j=0; j<dimN; j++)
       {
          sum_j +=  H[i*dimN+j] * x[j];
       }
       Hx[i] = sum_j;
    } 
}                           



double vecdot(int     dimN,
              double* x,
              double* y)
{
   double dotprod = 0.0;
   for (int i = 0; i < dimN; i++)
   {
      dotprod += x[i] * y[i];
   }

   return dotprod;
}

          
double vecmax(int     dimN,
              double* x)
{
    double max = - 1e+12;
    
    for (int i = 0; i < dimN; i++)
    {
        if (x[i] > max)
        {
           max = x[i];
        }
    }
    return max;
}


int argvecmax(int     dimN,
              double* x)
{
    double max = - 1e+12;
    int    i_max;
    for (int i = 0; i < dimN; i++)
    {
        if (x[i] > max)
        {
           max   = x[i];
           i_max = i;
        }
    }
    return i_max;
}