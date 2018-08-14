#include "l-bfgs.hpp"

L_BFGS::L_BFGS(int N, int stages)
{
   dimN = N;
   M    = stages;
   H0   = 1.0;

   /* Allocate memory for sk and yk for all stages */
   s = new double*[M];
   y = new double*[M];
   for (int imem = 0; imem < M; imem++)
   {
      s[imem] = new double[dimN];
      y[imem] = new double[dimN];
   }

   /* Allocate memory for rho's values */
   rho = new double[M];

   
}


L_BFGS::~L_BFGS()
{
   /* Deallocate memory */
   delete [] rho;
   for (int imem = 0; imem < M; imem++)
   {
      delete [] s[imem];
      delete [] y[imem];
   }
   delete [] s;
   delete [] y;
}



double L_BFGS::vecdot(double* x, double* y)
{
   double dotprod = 0.0;
   for (size_t i = 0; i < dimN; i++)
   {
      dotprod += x[i] * y[i];
   }
   
   return dotprod;
}


int L_BFGS::compute_step(int     k,
                         double* currgrad,
                         double* step)
{
   /* Initialize the step with steepest descent */
   for (int istep = 0; istep < dimN; istep++)
   {
      step[istep] = - currgrad[istep];
   }

   /* use steepest descent direction in first iteration */
   if (k == 0)
   {
      return 0;
   }

   int imin = k;
   int imax = (k-M);
   int imemory;
   double beta;
   double* alpha = new double[M];

   /* Loop backwards through lbfgs memory */
   for (int i = imax - 1; i >= imin; i--)
   {
      imemory = i % M;
      /* Compute alpha */
      alpha[imemory] = rho[imemory] * vecdot(s[imemory], step);
      /* Update the step */
      for (int istep = 0; istep < dimN; istep++)
      {
         step[istep] -= alpha[imemory] * y[imemory][istep];
      }
   }

   /* scale the step size by H0 */
   for (int istep = 0; istep < dimN; istep++)
   {
     step[istep] *= H0;
   }

  /* loop forwards through the l-bfgs memory */
  for (int i = imin; i < imax; i++)
  {
    imemory = i % M;
    /* Compute beta */
    beta = rho[imemory] * vecdot(y[imemory], step);
    /* Update the step */
    for (int istep = 0; istep < dimN; istep++)
    {
      step[istep] += s[imemory][istep] * (alpha[imemory] - beta);
    }
  }

  delete [] alpha;

   return 0;
}



int L_BFGS::update_memory(int     k,
                          double* xnew,
                          double* xold,
                          double* gradnew,
                          double* gradold)
{
   int imemory = k % M;
   /* Update y and s vector */
   for (int istep = 0; istep < dimN; istep++)
   {
     y[imemory][istep] = gradnew[istep] - gradold[istep];
     s[imemory][istep] = xnew[istep]    - xold[istep];
   }

   double yTs = vecdot(y[imemory], s[imemory]);
   if (yTs == 0.0) 
   {
     printf("  Warning: resetting yTs to 1.\n");
     yTs = 1.0;
   }

   rho[imemory] = 1. / yTs;
 
   double yTy = vecdot(y[imemory], y[imemory]);
   if (yTy == 0.) {
     // should print a warning here
     printf("  Warning: resetting yTy to 1.\n");
     yTy = 1.;
   }
   H0 = yTs / yTy;
  
   return 0;
}

