#include "HessianApprox.hpp"

HessianApprox::HessianApprox(int N)
{
  dimN = N;
}

HessianApprox::HessianApprox()
{
  dimN = 0;
}

HessianApprox::~HessianApprox(){}

double HessianApprox::vecdot(double* x, double* y)
{
   double dotprod = 0.0;
   for (int i = 0; i < dimN; i++)
   {
      dotprod += x[i] * y[i];
   }
   
   return dotprod;
}

int HessianApprox::matvec(double* H, 
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

    return 0;
}                           


L_BFGS::L_BFGS(int N, int stages) : HessianApprox(N)
{
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




int L_BFGS::compute_step(int     k,
                         double* currgrad,
                         double* step)
{
   int imemory;
   double beta;
   double* alpha = new double[M];
   int imax, imin;


   /* Initialize the step with steepest descent */
   for (int istep = 0; istep < dimN; istep++)
   {
      step[istep] = currgrad[istep];
   }


   /* Set range of the two-loop recursion */
   imax = k-1;
   if (k < M)
   {
     imin = 0;
   }
   else
   {
     imin = k - M;
   }

   /* Loop backwards through lbfgs memory */

   for (int i = imax; i >= imin; i--)
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
  for (int i = imin; i <= imax; i++)
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
   double yTy, yTs;
   int imemory = (k-1) % M ;

      /* Update BFGS memory for s, y */
   for (int istep = 0; istep < dimN; istep++)
   {
     y[imemory][istep] = gradnew[istep] - gradold[istep];
     s[imemory][istep] = xnew[istep]    - xold[istep];
   }

   /* Update rho and H0 */
   yTs = vecdot(y[imemory], s[imemory]);
   yTy = vecdot(y[imemory], y[imemory]);
   if (yTs == 0.0) 
   {
     printf("  Warning: resetting yTs to 1.\n");
     yTs = 1.0;
   }
   if (yTy == 0.0) 
   {
     printf("  Warning: resetting yTy to 1.\n");
     yTy = 1.;
   }
   rho[imemory] = 1. / yTs;
   H0 = yTs / yTy;
  
   return 0;
}



BFGS::BFGS(int N) : HessianApprox(N)
{
    Hessian = new double[N*N];
    setIdentity();

    y = new double[N];
    s = new double[N];
}

int BFGS::setIdentity()
{
    for (int i = 0; i<dimN; i++)
    {
      for (int j = 0; j<dimN; j++)
      {
          if (i==j) Hessian[i*dimN+j] = 1.0;
          else      Hessian[i*dimN+j] = 0.0;
      }
    }
    return 0;
}

BFGS::~BFGS()
{
  delete [] Hessian;
  delete [] y; 
  delete [] s; 
}


int BFGS::update_memory(int     k,
                        double* xnew,
                        double* xold,
                        double* gradnew,
                        double* gradold)
{
    /* Update BFGS memory for s, y */
    for (int istep = 0; istep < dimN; istep++)
    {
      y[istep] = gradnew[istep] - gradold[istep];
      s[istep] = xnew[istep]    - xold[istep];
    }
 
  
    return 0;
}

int BFGS::compute_step(int     k, 
                       double* currgrad, 
                       double* step)
{
    double yTy, yTs, H0;
    double sum, sum1, sum2, second, yTHy;
    double rho;

    /* Check curvature conditoin */
    yTs = vecdot(y, s);
    if (yTs < 1e-12) 
    {
      printf("  Warning: Curvature condition not satisfied %1.14e \n", yTs);
      setIdentity();

      matvec(Hessian, currgrad, step);

      return 0;
    }
   
    /* Scale first Hessian approximation */
    yTy = vecdot(y, y);
    if (k == 1)
    {
      H0  = yTs / yTy;
      for (int i=0; i<dimN; i++)
      {
          Hessian[i*dimN+i] = H0;
      }
    }

    rho = 1. / yTs;

    /* Scalar yTHy */
    yTHy = 0.0;
    for (int i = 0; i<dimN; i++)
    {
        sum = 0.0;
        for (int j = 0; j<dimN; j++)
        {
            sum += Hessian[i*dimN+j] * y[j];
        }
        yTHy += y[i] * sum;
    }
    second = (yTs + yTHy) / pow(yTs,2);
        
    /* Updating the lower triangular part */
    for (int i = 0; i<dimN; i++)
    {
      for (int j=0; j <= i; j++)
      {
        sum1 = 0.0;
        sum2 = 0.0;
        for (int m=0; m<dimN; m++)
        {
          if (m<i)
          {
            sum1 += Hessian[m*dimN + i ]*y[m];  // lower half -> exploit symmetry of H!
            sum2 += Hessian[m*dimN + j ]*y[m];  // lower half -> exploit symmetry of H!
          } 
          else
          {
            sum1 += Hessian[i*dimN + m ]*y[m];
            sum2 += Hessian[j*dimN + m ]*y[m];
          }     
        }
        Hessian[i*dimN + j] += second * s[i]*s[j] - rho * ( s[j]*sum1  + s[i]*sum2 );
      }
    }
    
    /* Fill the upper half with symmetrie */
    for (int i = 0; i<dimN; i++)
    {
        for (int j=0; j<i; j++)
        {
          Hessian[j*dimN+i] = Hessian[i*dimN+j];
        }
    }

    /* Compute the step */
    matvec(Hessian, currgrad, step);

    return 0;
}
