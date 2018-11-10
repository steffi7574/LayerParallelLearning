#include "linalg.hpp"
#include "hessianApprox.hpp"

HessianApprox::HessianApprox(){}
HessianApprox::~HessianApprox(){}


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
      for (int i = 0; i < dimN; i++)
      {
          s[imem][i] = 0.0;
          y[imem][i] = 0.0;
      }
   }

   /* Allocate memory for rho's values */
   rho = new double[M];
   for (int i = 0; i < M; i++)
   {
       rho[i] = 0.0;
   }
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




void L_BFGS::computeDescentDir(int     iter,
                               double* currgrad,
                               double* descdir)
{
   int imemory;
   double beta;
   double* alpha = new double[M];
   int imax, imin;


   /* Initialize the descdir with steepest descent */
   for (int idir = 0; idir < dimN; idir++)
   {
      descdir[idir] = currgrad[idir];
   }


   /* Set range of the two-loop recursion */
   imax = iter-1;
   if (iter < M)
   {
     imin = 0;
   }
   else
   {
     imin = iter - M;
   }

   /* Loop backwards through lbfgs memory */
   for (int i = imax; i >= imin; i--)
   {
      imemory = i % M;
      /* Compute alpha */
      alpha[imemory] = rho[imemory] * vecdot(dimN, s[imemory], descdir);
      /* Update the descdir */
      for (int idir = 0; idir < dimN; idir++)
      {
         descdir[idir] -= alpha[imemory] * y[imemory][idir];
      }
   }

   /* scale the descdir by H0 */
   for (int idir = 0; idir < dimN; idir++)
   {
     descdir[idir] *= H0;
   }

  /* loop forwards through the l-bfgs memory */
  for (int i = imin; i <= imax; i++)
  {
    imemory = i % M;
    /* Compute beta */
    beta = rho[imemory] * vecdot(dimN, y[imemory], descdir);
    /* Update the descdir */
    for (int idir = 0; idir < dimN; idir++)
    {
      descdir[idir] += s[imemory][idir] * (alpha[imemory] - beta);
    }
  }

  delete [] alpha;

}



void L_BFGS::updateMemory(int     iter,
                          double* xnew,
                          double* xold,
                          double* gradnew,
                          double* gradold)
{

  /* Update lbfgs memory if iter > 0 */
  if (iter > 0) 
  {
     double yTy, yTs;
     int imemory = (iter-1) % M ;

        /* Update BFGS memory for s, y */
     for (int idir = 0; idir < dimN; idir++)
     {
       y[imemory][idir] = gradnew[idir] - gradold[idir];
       s[imemory][idir] = xnew[idir]    - xold[idir];
     }

     /* Update rho and H0 */
     yTs = vecdot(dimN, y[imemory], s[imemory]);
     yTy = vecdot(dimN, y[imemory], y[imemory]);
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
  }
}



BFGS::BFGS(int N) 
{
    dimN = N;

    Hessian = new double[N*N];
    setIdentity();

    y = new double[N];
    s = new double[N];

    Hy = new double[N];
    A  = new double[N*N];
    B  = new double[N*N];
}

void BFGS::setIdentity()
{
    for (int i = 0; i<dimN; i++)
    {
      for (int j = 0; j<dimN; j++)
      {
          if (i==j) Hessian[i*dimN+j] = 1.0;
          else      Hessian[i*dimN+j] = 0.0;
      }
    }
}

BFGS::~BFGS()
{
  delete [] Hessian;
  delete [] y; 
  delete [] s; 
  delete [] A; 
  delete [] B; 
  delete [] Hy; 
}


void BFGS::updateMemory(int     iter,
                        double* xnew,
                        double* xold,
                        double* gradnew,
                        double* gradold)
{
    /* Update BFGS memory for s, y */
    for (int idir = 0; idir < dimN; idir++)
    {
      y[idir] = gradnew[idir] - gradold[idir];
      s[idir] = xnew[idir]    - xold[idir];
    }
}

void BFGS::computeDescentDir(int     iter, 
                             double* currgrad, 
                             double* descdir)
{
    double yTy, yTs, H0;
    double b, rho;

    /* Steepest descent in first iteration */
    if (iter == 0)
    {
      setIdentity();
      matvec(dimN, Hessian, currgrad, descdir);
      return;
    }

    /* Check curvature conditoin */
    yTs = vecdot(dimN, y, s);
    if ( yTs < 1e-12) 
    {
      printf(" Warning: Curvature condition not satisfied %1.14e \n", yTs);
      setIdentity();
    }
    else
    {
      /* Scale first Hessian approximation */
      yTy = vecdot(dimN, y, y);
      if (iter == 1)
      {
        H0  = yTs / yTy;
        for (int i=0; i<dimN; i++)
        {
            Hessian[i*dimN+i] = H0;
        }
      }

      /** BFGS Update for H  (Noceda, Wright, Chapter 6.1)
       *  H_new  = H + \rho( B - (A+A'))
       * where B = (1.0 + \rho * y'Hy) * ss'
       *       A = Hys'
       */

      /* Compute A = Hys' */
      matvec(dimN, Hessian, y, Hy);
      vecvecT(dimN, Hy, s, A);

      /* scalar 1 + rho y'Hy */
      rho = 1. / yTs;
      b   = 1.0 + rho * vecdot(dimN, y, Hy);

      /* Compute B */
      vecvecT(dimN, s, s, B);

      /* H += rho * (b*B - (A+A')) */
      for (int i=0; i<dimN; i++)
      {
        for (int j=0; j<dimN; j++)
        {
           Hessian[i*dimN+j] += rho * ( b * B[i*dimN+j] - A[i*dimN+j] - A[j*dimN+i] ) ;
        }
      } 

    }

    /* Compute the descdir */
    matvec(dimN, Hessian, currgrad, descdir);
}

Identity::Identity(int N) 
{
  dimN = N;
}

Identity::~Identity(){}

void Identity::updateMemory(int     iter,
                            double* xnew,
                            double* xold,
                            double* gradnew,
                            double* gradold){}



void Identity::computeDescentDir(int     iter, 
                                 double* currgrad, 
                                 double* descdir)
{
  /* Steepest descent */
  for (int i = 0; i<dimN; i++)
  {
    descdir[i] = currgrad[i];
  }
}                           


