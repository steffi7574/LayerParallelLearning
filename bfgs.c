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
bfgs(int     N,
     double *design,
     double *design_old,
     double *gradient,
     double *gradient_old,
     double *H)
{
   
   double* s      = (double*)malloc(N*sizeof(double));
   double* y      = (double*)malloc(N*sizeof(double));
   double* first  = (double*)malloc(N*N*sizeof(double));
   double* second = (double*)malloc(N*N*sizeof(double));
   double* third  = (double*)malloc(N*N*sizeof(double));
   double* Hy     = (double*)malloc(N*sizeof(double));
   double  rho, scalar_first;



        for (int i = 0; i < N; i++)
        {
            /* Update sk and yk for bfgs */
            s[i] = design[i] - design_old[i];
            y[i] = gradient[i] - gradient_old[i];
        }
 

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
   free(s);
   free(y);




    // double sum, sum1, sum2, second, yTHy;
    // /* Scalar yTHy */
    // yTHy = 0.0;
    // for (int i = 0; i<dimN; i++)
    // {
    //     sum = 0.0;
    //     for (int j = 0; j<dimN; j++)
    //     {
    //         sum += Hessian[i*dimN+j] * y[j];
    //     }
    //     yTHy += y[i] * sum;
    // }
    // second = (yTs + yTHy) / pow(yTs,2);
        
    // /* Updating the lower triangular part */
    // for (int i = 0; i<dimN; i++)
    // {
    //   for (int j=0; j <= i; j++)
    //   {
    //     sum1 = 0.0;
    //     sum2 = 0.0;
    //     for (int m=0; m<dimN; m++)
    //     {
    //       if (m<i)
    //       {
    //         sum1 += Hessian[m*dimN + i ]*y[m];  // lower half -> exploit symmetry of H!
    //         sum2 += Hessian[m*dimN + j ]*y[m];  // lower half -> exploit symmetry of H!
    //       } 
    //       else
    //       {
    //         sum1 += Hessian[i*dimN + m ]*y[m];
    //         sum2 += Hessian[j*dimN + m ]*y[m];
    //       }     
    //     }
    //     Hessian[i*dimN + j] += second * s[i]*s[j] - rho * ( s[j]*sum1  + s[i]*sum2 );
    //   }
    // }
    
    // /* Fill the upper half with symmetrie */
    // for (int i = 0; i<dimN; i++)
    // {
    //     for (int j=0; j<i; j++)
    //     {
    //       Hessian[j*dimN+i] = Hessian[i*dimN+j];
    //     }
    // }



   return 0;
}         
