#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lib.h"


RealReverse 
max(RealReverse a,
    RealReverse b)
{
   RealReverse max = a;
   if (a < b)
   {
      max = b;
   }
   return max;
}


RealReverse 
sigma(RealReverse x)
{
   RealReverse sigma;

   /* ReLU activation function */
//    sigma = max(0,x);

   /* tanh activation */
   sigma = tanh(x);

   return sigma;
}

RealReverse
sigma_diff(RealReverse x)
{
    RealReverse ddx;
    RealReverse tmp;

    /* ReLu actionvation */
    // if (max(0,x) > 0)
        // ddx = 1.0;
    // else 
        // ddx = 0.0;

    /* tanh activation */
    tmp = tanh(x);
    ddx = 1. - tmp * tmp;

    return ddx;
}


int
take_step(RealReverse* Y,
          RealReverse* theta,
          int     ts,
          double  dt,
          int    *batch,
          int     nbatch,
          int     nchannels, 
          int     parabolic)
{
   /* Element Y_id stored in Y[id * nf, ..., ,(id+1)*nf -1] */
   RealReverse sum;
   int    th_idx;
   int    batch_id;
   RealReverse *update = (RealReverse*)malloc(nchannels * sizeof(RealReverse));

   /* iterate over all batch elements */ 
   for (int i = 0; i < nbatch; i++)
   {
      batch_id = batch[i];

      /* Iterate over all channels */
      for (int ichannel = 0; ichannel < nchannels; ichannel++)
      {
         /* Apply weights */
         sum = 0.0;
         for (int jchannel = 0; jchannel < nchannels; jchannel++)
         {
            th_idx = ts * ( nchannels * nchannels + 1) + jchannel * nchannels + ichannel;
            sum += theta[th_idx] * Y[batch_id * nchannels + jchannel];
         }
         update[ichannel] = sum;

         /* Apply activation */
         update[ichannel] = sigma(update[ichannel]);

         /* Apply bias */
         th_idx = ts * (nchannels * nchannels + 1) + nchannels*nchannels;
         update[ichannel] += theta[th_idx];
      }

      /* Apply transposed weights, if necessary, and update */
      for (int ichannel = 0; ichannel < nchannels; ichannel++)
      {
         sum = 0.0;
         if (parabolic)
         {
            for (int jchannel = 0; jchannel < nchannels; jchannel++)
            {
               th_idx = ts * (nchannels * nchannels + 1) + jchannel * nchannels + ichannel;
               sum += theta[th_idx] * update[jchannel]; 
            }
         } 
         else
         {
            sum = update[ichannel];
         }

     
         int idx = batch_id * nchannels + ichannel;
         Y[idx] += dt * sum;
 
        //  if (batch_id == 0) printf("upd %f * %1.14e ", dt, sum);
        //  if (batch_id == 0) printf("Y[%d] = %1.14e\n", idx, Y[idx] );
      }
   }      

   free(update);
   return 0;
}


RealReverse
loss(RealReverse  *Y,
     double       *Ytarget,
     int          *batch,
     int           nbatch,
     int           nchannels)
{
   int    idx;
   int    batch_id;

   /* Loop over batch elements */
   RealReverse objective = 0.0;
   for (int ibatch = 0; ibatch < nbatch; ibatch ++)
   {
       /* Get batch_id */
       batch_id = batch[ibatch];

       /* Add to objective function */
       for (int ichannel = 0; ichannel < nchannels; ichannel++)
       {
          idx = batch_id * nchannels + ichannel;
          objective += 1./2. * (Y[idx] - Ytarget[idx]) * (Y[idx] - Ytarget[idx]);
        //   printf("OBJ LOSS idx %d\n", idx );
       }
   }

   return objective;
}

RealReverse
regularization(RealReverse* theta,
               int          ts,
               double       dt,
               int          ntime,
               int          nchannels)
{
   RealReverse relax = 0.0;
   int         idx, idx1;

    /* K(theta)-part */
    for (int ichannel = 0; ichannel < nchannels; ichannel++)
    {
        for (int jchannel = 0; jchannel < nchannels; jchannel++)
        {
            idx  = ts       * (nchannels * nchannels + 1) + ichannel *  nchannels + jchannel;
            idx1 = ( ts+1 ) * (nchannels * nchannels + 1) + ichannel *  nchannels + jchannel;
  
            relax += 1./2.* theta[idx] * theta[idx];
            if (ts < ntime - 1) 
            {
               relax += 1./2. * (theta[idx1] - theta[idx]) * (theta[idx1] - theta  [idx]) / dt;
            } 
        }
    }

    /* b(theta)-part */
    idx  =   ts     * ( nchannels * nchannels + 1) + nchannels*nchannels;
    relax += 1./2. * theta[idx] * theta[idx];
    if (ts < ntime - 1)
    {
        idx1 = ( ts+1 ) * ( nchannels * nchannels + 1) + nchannels*nchannels;
        relax += 1./2. * (theta[idx1] - theta[idx]) * (theta[idx1] - theta[idx]) / dt;
    }

    return relax;
}        


template <typename myType> 
int
read_data(char *filename, myType *var, int size)
{

   FILE   *file;
   double  tmp;
   int     i;

   /* open file */
   file = fopen(filename, "r");

   /* Read data */
   if (file == NULL)
   {
      printf("Can't open %s \n", filename);
      exit(1);
   }
   for ( i = 0; i < size; i++)
   {
      fscanf(file, "%lf", &tmp);
      var[i] = tmp;
   }

   /* close file */
   fclose(file);

   return 0;
}

template <typename myType>
int
write_data(char *filename, myType *var, int size)
{
   FILE *file;
   int i;

   /* open file */
   file = fopen(filename, "w");

   /* Read data */
   if (file == NULL)
   {
      printf("Can't open %s \n", filename);
      exit(1);
   }
   printf("Writing file %s\n", filename);
   for ( i = 0; i < size; i++)
   {
      fprintf(file, "%1.14e\n", getValue(var[i]));
   }

   /* close file */
   fclose(file);

   return 0;

}

/* Explicit instantiation of the template functions */
template int read_data<double>(char *filename, double *var, int size);
template int read_data<RealReverse>(char *filename, RealReverse *var, int size);
template int write_data<double>(char *filename, double *var, int size);
template int write_data<RealReverse>(char *filename, RealReverse *var, int size);

double 
getValue(double value)
{
    return value;
}

double 
getValue(RealReverse value)
{
    return value.getValue();
}


