#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double 
max(double a,
    double b)
{
   double max = a;
   if (a < b)
   {
      max = b;
   }
   return max;
}


double 
sigma(double x)
{
   double sigma;

   /* ReLU activation function */
//    sigma = max(0,x);

   /* tanh activation */
   sigma = tanh(x);

   return sigma;
}


int
take_step(double* Y,
          double* theta,
          int     ts,
          double  dt,
          int    *batch,
          int     nbatch,
          int     nchannels, 
          int     parabolic)
{
   /* Element Y_id stored in Y[id * nf, ..., ,(id+1)*nf -1] */
   double sum;
   int    th_idx;
   int    batch_id;
   double *update = (double*)malloc(nchannels * sizeof(double));

   /* iterate over all batch elements */ 
   for (int i = 0; i < nbatch; i++)
   {
      batch_id = batch[i];

      /* Iterate over all channels of that batch element */
      for (int ichannel = 0; ichannel < nchannels; ichannel++)
      {
         /* Apply weights */
         sum = 0.0;
         for (int jchannel = 0; jchannel < nchannels; jchannel++)
         {
            th_idx = ts * ( nchannels * nchannels + 1) + ichannel * nchannels + jchannel;
            sum += theta[th_idx] * Y[batch_id * nchannels + jchannel];
         }
         update[ichannel] = sum;


         /* Apply activation */
         update[ichannel] = sigma(update[ichannel]);

         /* Apply bias */
         update[ichannel] += theta[ts * nchannels * nchannels];

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


int
read_data(char *filename, double *var, int size)
{

   FILE *file;
   int   i;

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
      fscanf(file, "%lf", &(var[i]));
   }

   /* close file */
   fclose(file);

   return 0;
}

int
write_data(char *filename, double *var, int size)
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
      fprintf(file, "%1.14e\n", var[i]);
   }

   /* close file */
   fclose(file);

   return 0;

}

double  
loss(double*  Y,
     double*  Ytarget,
     int     *batch,
     int      nbatch,
     int      nchannels)
{
   int    idx;
   int    batch_id;

   /* Loop over batch elements */
   double objective = 0.0;
   for (int ibatch = 0; ibatch < nbatch; ibatch ++)
   {
       /* Get batch_id */
       batch_id = batch[ibatch];

       /* Add to objective function */
       for (int ichannel = 0; ichannel < nchannels; ichannel++)
       {
          idx = batch_id * nchannels + ichannel;
          objective += 1./2. * (Y[idx] - Ytarget[idx]) * (Y[idx] - Ytarget[idx]);
       }
   }

   return objective;
}

double
regularization(double* theta,
               int     ts,
               double  dt,
               int     ntime,
               int     nchannels)
{
   double relax = 0.0;
   int idx, idx1;

   for (int ichannel = 0; ichannel < nchannels; ichannel++)
   {
      /* K(theta)-part */
      for (int jchannel = 0; jchannel < nchannels; jchannel++)
      {
         idx  = ts       * (nchannels * nchannels + 1) + ichannel * nchannels + jchannel;
         idx1 = ( ts+1 ) * (nchannels * nchannels + 1) + ichannel * nchannels + jchannel;

         relax += theta[idx] * theta[idx];

         if (ts < ntime - 1) 
         {
            relax += (theta[idx1] - theta[idx]) * (theta[idx1] - theta[idx]) / dt;
         }
      }

      /* b(theta)-part */
      idx  = ( ts+1 ) * nchannels * nchannels;
      idx1 = ( ts+2 ) * nchannels * nchannels;
      relax += theta[idx] * theta[idx];
      relax += (theta[idx1] - theta[idx]) * (theta[idx1] - theta[idx]) / dt;
   }

   return relax;
}        