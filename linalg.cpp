#include "linalg.hpp"

void matvec(int dimN,
            MyReal* H, 
            MyReal* x,
            MyReal* Hx)
{
    MyReal sum_j;

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



MyReal vecdot(int     dimN,
              MyReal* x,
              MyReal* y)
{
   MyReal dotprod = 0.0;
   for (int i = 0; i < dimN; i++)
   {
      dotprod += x[i] * y[i];
   }
   return dotprod;
}

          
MyReal vecmax(int     dimN,
              MyReal* x)
{
    MyReal max = - 1e+12;
    
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
              MyReal* x)
{
    MyReal max = - 1e+12;
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


MyReal vec_normsq(int    dimN,
                  MyReal *x)
{
    MyReal norm = 0.0;
    for (int i = 0; i<dimN; i++)
    {
        norm += pow(x[i],2);
    }

    return norm;
}

int vec_copy(int N, 
             MyReal* u, 
             MyReal* u_copy)
{
    for (int i=0; i<N; i++)
    {
        u_copy[i] = u[i];
    }

    return 0;
}

void vecvecT(int N,
             MyReal* x,
             MyReal* y,
             MyReal* XYT)
{
   for (int i=0; i<N; i++)
   {
      for (int j=0; j<N; j++)
      {
         XYT[i*N+j] = x[i]*y[j];
      }
   }
}

