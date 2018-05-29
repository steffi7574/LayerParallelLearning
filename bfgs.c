#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "bfgs.h"

int 
set_identity(int N,
             double* A)
{
   for (int i=0; i<N; i++)
   {
      for (int j=0; j<N; j++)
      {
         if (i==j) A[i*N+j] = 1.0;
         else      A[i*N+j] = 0.0;
      }
   }
   return 0;
}

double
vecT_vec_product(int     N,
                 double *x, 
                 double *y)
{
   double prd = 0.0;

   for (int i=0; i < N ; i++)
   {
      prd += x[i] * y[i];
   }

   return prd;
}              


int 
vec_vecT_product(int N,
                     double *x,
                     double *y,
                     double *A)
{
   for (int i=0; i<N; i++)
   {
      for (int j=0; j<N; j++)
      {
         A[i*N+j] = x[i]*y[j];
      }
   }

   return 0;
}       


/* return y = Ax */
int
mat_vec_product(int N,
                double *A,
                double *x,
                double *y)
{
   double sum_j;

   for (int i=0; i<N; i++)
   {
      sum_j = 0.0;
      for (int j=0; j<N; j++)
      {
         sum_j +=  A[i*N+j] * x[j];
      }
      y[i] = sum_j;
   } 

   return 0;
}             

int
vecT_mat_product(int N,
                 double *x,
                 double *A,
                 double *y)
{
   double sum_j;
   for (int i=0; i<N; i++)
   {
      sum_j = 0.0;
      for (int j=0; j<N; j++)
      {
         sum_j +=  x[j] * A[j*N+i];
      }
      y[i] = sum_j;
   }

   return 0;
}              

int
bfgs_update(int     N,
            double *s,
            double *y,
            double *H)

{
   double* first  = (double*)malloc(N*N*sizeof(double));
   double* second = (double*)malloc(N*N*sizeof(double));
   double* third  = (double*)malloc(N*N*sizeof(double));
   double* Hy     = (double*)malloc(N*sizeof(double));
   double  rho, scalar_first;

   /* Calculate curvature condition */
   rho = vecT_vec_product(N, y, s);

   /* BFGS Update for H */
   if (rho > 1e-12)
   {
      /* First term: scalar */
      mat_vec_product(N, H, y, Hy);
      scalar_first = vecT_vec_product(N, y, Hy);
      scalar_first = (scalar_first + rho) / pow(rho,2);

      /* First term ss^T*/
      vec_vecT_product(N, s, s, first);

      /* Second term Hys^T*/
      vec_vecT_product(N, Hy, s, second);
      
      /* Third term sy^TH */
      vecT_mat_product(N, y, H, Hy);
      vec_vecT_product(N, s, Hy, third);      

      /* Sum up */
      for (int i=0; i<N; i++)
      {
         for (int j=0; j<N; j++)
         {
            H[i*N+j] += scalar_first * first[i*N+j] - 1./rho * ( second[i*N+j] + third[i*N+j] );
         }
      }

   }
   else
   {
      set_identity(N, H);
      printf("Curv %1.14e, ", rho);
      printf("Setting H to identity\n");
   }

   /* Free memory */
   free(first);
   free(second);
   free(third);
   free(Hy);


   return 0;
}         
