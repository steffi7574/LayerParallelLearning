#include "hessianApprox.hpp"

HessianApprox::HessianApprox(MPI_Comm comm)
{
  dimN    = 0;
  MPIcomm = comm;
}
HessianApprox::~HessianApprox(){}


L_BFGS::L_BFGS(MPI_Comm comm, int N, int stages) : HessianApprox(comm)
{
   dimN = N;
   M    = stages;
   H0   = 1.0;

   /* Allocate memory for sk and yk for all stages */
   s = new MyReal*[M];
   y = new MyReal*[M];
   for (int imem = 0; imem < M; imem++)
   {
      s[imem] = new MyReal[dimN];
      y[imem] = new MyReal[dimN];
      for (int i = 0; i < dimN; i++)
      {
          s[imem][i] = 0.0;
          y[imem][i] = 0.0;
      }
   }

   /* Allocate memory for rho's values */
   rho = new MyReal[M];
   for (int i = 0; i < M; i++)
   {
       rho[i] = 0.0;
   }

   /* Allocate memory for storing design at previous iteration */
   design_old   = new MyReal[dimN];
   gradient_old = new MyReal[dimN];
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

   delete [] design_old;
   delete [] gradient_old;
}




void L_BFGS::computeAscentDir(int     iter,
                              MyReal* gradient,
                              MyReal* ascentdir)
{
   int imemory;
   MyReal beta;
   MyReal* alpha = new MyReal[M];
   int imax, imin;


   /* Initialize the ascentdir with steepest descent */
   for (int idir = 0; idir < dimN; idir++)
   {
      ascentdir[idir] = gradient[idir];
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
      alpha[imemory] = rho[imemory] * vecdot_par(dimN, s[imemory], ascentdir, MPIcomm);
      /* Update the ascentdir */
      for (int idir = 0; idir < dimN; idir++)
      {
         ascentdir[idir] -= alpha[imemory] * y[imemory][idir];
      }
   }

   /* scale the ascentdir by H0 */
   for (int idir = 0; idir < dimN; idir++)
   {
     ascentdir[idir] *= H0;
   }

  /* loop forwards through the l-bfgs memory */
  for (int i = imin; i <= imax; i++)
  {
    imemory = i % M;
    /* Compute beta */
    beta = rho[imemory] * vecdot_par(dimN, y[imemory], ascentdir, MPIcomm);
    /* Update the ascentdir */
    for (int idir = 0; idir < dimN; idir++)
    {
      ascentdir[idir] += s[imemory][idir] * (alpha[imemory] - beta);
    }
  }

  delete [] alpha;

}



void L_BFGS::updateMemory(int     iter,
                          MyReal* design,
                          MyReal* gradient)
{

  /* Update lbfgs memory only if iter > 0 */
  if (iter > 0) 
  {
     MyReal yTy, yTs;

     /* Get storing state */
     int imemory = (iter-1) % M ;

     /* Update BFGS memory for s, y */
     for (int idir = 0; idir < dimN; idir++)
     {
       y[imemory][idir] = gradient[idir] - gradient_old[idir];
       s[imemory][idir] = design[idir]   - design_old[idir];
     }

     /* Update rho and H0 */
     yTs = vecdot_par(dimN, y[imemory], s[imemory], MPIcomm);
     yTy = vecdot_par(dimN, y[imemory], y[imemory], MPIcomm);
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

   /* Update old design and gradient */
   vec_copy(dimN, design,   design_old);
   vec_copy(dimN, gradient, gradient_old);
}



BFGS::BFGS(MPI_Comm comm, int N) : HessianApprox(comm)
{
    dimN = N;

    Hessian = new MyReal[N*N];
    setIdentity();

    y = new MyReal[N];
    s = new MyReal[N];

    Hy = new MyReal[N];
    A  = new MyReal[N*N];
    B  = new MyReal[N*N];

    /* Allocate memory for storing design at previous iteration */
    design_old   = new MyReal[dimN];
    gradient_old = new MyReal[dimN];

    /* Sanity check */
    int size;
    MPI_Comm_size(MPIcomm, &size);
    if (size > 1) printf("\n\n WARNING: Parallel BFGS not implemented.\n BFGS updates will be LOCAL to each processor -> block-BFGS. \n\n");
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
  delete [] design_old;
  delete [] gradient_old;
}


void BFGS::updateMemory(int     iter,
                        MyReal* design,
                        MyReal* gradient)
{
    /* Update BFGS memory for s, y */
    for (int idir = 0; idir < dimN; idir++)
    {
      y[idir] = gradient[idir] - gradient_old[idir];
      s[idir] = design[idir]   - design_old[idir];
    }
}

void BFGS::computeAscentDir(int     iter, 
                             MyReal* gradient, 
                             MyReal* ascentdir)
{
    MyReal yTy, yTs, H0;
    MyReal b, rho;

    /* Steepest descent in first iteration */
    if (iter == 0)
    {
      setIdentity();
      matvec(dimN, Hessian, gradient, ascentdir);
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

    /* Compute the ascentdir */
    matvec(dimN, Hessian, gradient, ascentdir);
}

Identity::Identity(MPI_Comm comm, int N) : HessianApprox(comm)
{
  dimN = N;
}

Identity::~Identity(){}

void Identity::updateMemory(int     iter,
                            MyReal* design,
                            MyReal* gradient) {}



void Identity::computeAscentDir(int     iter, 
                                 MyReal* gradient, 
                                 MyReal* ascentdir)
{
  /*  Steepest descent */
  for (int i = 0; i<dimN; i++)
  {
    ascentdir[i] = gradient[i];
  }
}                           


