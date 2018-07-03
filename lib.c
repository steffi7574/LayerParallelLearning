#include "lib.h"


template <typename myDouble>
myDouble 
maximum(myDouble *a,
        int       size_t)
{
   myDouble max = a[0];
   
   for (int i = 1; i < size_t; i++)
   {
       if (a[i] > max)
       {
          max = a[i];
       }
   }

   return max;
}


template <typename myDouble>
myDouble 
sigma(myDouble x)
{
   myDouble sigma;

   /* ReLU activation function */
//    sigma = max(0,x);

   /* tanh activation */
   sigma = tanh(x);

   return sigma;
}

double
sigma_diff(double x)
{
    double ddx;
    double tmp;

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


template <typename myDouble>
int
take_step(myDouble* Y,
          myDouble* theta,
          int     ts,
          double  dt,
          int    *batch,
          int     nbatch,
          int     nchannels, 
          int     parabolic)
{
   /* Element Y_id stored in Y[id * nf, ..., ,(id+1)*nf -1] */
   myDouble sum;
   int    th_idx;
   int    batch_id;
   myDouble *update = (myDouble*)malloc(nchannels * sizeof(myDouble));

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

         /* Apply bias */
         th_idx = ts * (nchannels * nchannels + 1) + nchannels*nchannels;
         update[ichannel] += theta[th_idx];

         /* Apply nonlinear activation */
         update[ichannel] = sigma(update[ichannel]);
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


template <typename myDouble>
myDouble
loss(myDouble     *Y,
     double       *Target,
     int          *batch,
     int           nbatch,
     myDouble     *classW,
     myDouble     *classMu,
     int           nclasses,
     int           nchannels)
{
   myDouble loss; 
   myDouble normalization, tmp;
   int batch_id, weight_id, y_id, target_id;
   myDouble* YW_batch = new myDouble [nclasses];


   loss = 0.0;
   /* Loop over batch elements */
   for (int ibatch = 0; ibatch < nbatch; ibatch ++)
   {
       /* Get batch_id */
       batch_id = batch[ibatch];

        /* Apply the classification weights YW */
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            YW_batch[iclass] = 0.0;

            for (int ichannel = 0; ichannel < nchannels; ichannel++)
            {
                y_id      = batch_id * nchannels + ichannel;
                weight_id = iclass   * nchannels + ichannel;
                YW_batch[iclass] += Y[y_id] * classW[weight_id];
            }
        }

        /* Add the classification bias YW + mu */
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            YW_batch[iclass] += classMu[iclass];
        }

        /* Pointwise Normalization YW + mu - max(classes) */
        normalization = maximum (YW_batch, nclasses);
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            YW_batch[iclass] -= normalization;
        }

        /* First term: sum (C.*YW) */
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            target_id = batch_id * nclasses + iclass;
            loss -= Target[target_id] * YW_batch[iclass];
        }

        /* Second term: log(sum(exp.(YW))) */
        tmp = 0.0;
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            tmp += exp(YW_batch[iclass]);
        }
        loss += log(tmp);



       
        /* Evaluate loss */
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            target_id = batch_id * nclasses + iclass;
            loss += 1./2. * (YW_batch[iclass] - Target[target_id]) * (YW_batch[iclass] - Target[target_id]);
        }
   }

   delete [] YW_batch;

   return loss;
}


template <typename myDouble>
myDouble
regularization_class(myDouble *classW, 
                     myDouble *classMu, 
                     int       nclasses, 
                     int       nchannels)
{
    myDouble relax = 0.0;
    int      idx;

    /* W-part */
    for (idx = 0; idx < nclasses * nchannels; idx++)
    {
        relax += 1./2. * (classW[idx] * classW[idx]);
    }

    /* mu-part */
    for (idx = 0; idx < nclasses; idx++)
    {
        relax += 1./2. * (classMu[idx] * classMu[idx]);
    }

    return relax;
}


template <typename myDouble>
myDouble
regularization_theta(myDouble* theta,
                     int          ts,
                     double       dt,
                     int          ntime,
                     int          nchannels)
{
   myDouble relax = 0.0;
   int      idx, idx1;

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


int
gradient_allreduce(braid_App app, 
                   MPI_Comm comm)
{

   int ntheta   = (app->nchannels * app->nchannels + 1) * app->ntimes;
   int nclassW  =  app->nchannels * app->nclasses;
   int nclassmu =  app->nclasses;

   double *theta_grad   = (double*) malloc(ntheta   *sizeof(double));
   double *classW_grad  = (double*) malloc(nclassW  *sizeof(double));
   double *classmu_grad = (double*) malloc(nclassmu *sizeof(double));
   for (int i = 0; i<ntheta; i++)
   {
       theta_grad[i] = app->theta_grad[i];
   }
   for (int i = 0; i < nclassW; i++)
   {
       classW_grad[i] = app->classW_grad[i];
   }
   for (int i = 0; i < nclassmu; i++)
   {
       classmu_grad[i] = app->classMu_grad[i];
   }

   /* Collect sensitivities from all time-processors */
   MPI_Allreduce(theta_grad, app->theta_grad, ntheta, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(classW_grad, app->classW_grad, nclassW, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(classmu_grad, app->classMu_grad, nclassmu, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   free(theta_grad);
   free(classW_grad);
   free(classmu_grad);

   return 0;
}                  




int
gradient_norm(braid_App app,
              double   *theta_gnorm_prt,
              double   *class_gnorm_prt)

{
    double ntheta = (app->nchannels * app->nchannels + 1) * app->ntimes;
    int nclassW   =  app->nchannels * app->nclasses;
    int nclassmu  =  app->nclasses;
    double theta_gnorm, class_gnorm;

    /* Norm of gradient */
    theta_gnorm = 0.0;
    class_gnorm = 0.0;
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta_gnorm += pow(getValue(app->theta_grad[itheta]), 2);
    }
    for (int i = 0; i < nclassW; i++)
    {
        class_gnorm += pow(getValue(app->classW_grad[i]),2);
    }
    for (int i = 0; i < nclassmu; i++)
    {
        class_gnorm += pow(getValue(app->classMu_grad[i]),2);
    }
    theta_gnorm = sqrt(theta_gnorm);
    class_gnorm = sqrt(class_gnorm);

    *theta_gnorm_prt = theta_gnorm;
    *class_gnorm_prt = class_gnorm;
    
    return 0;
}
    
   
int 
update_theta(braid_App app, 
             double    stepsize,
             double   *direction)
{
    int ntheta = (app->nchannels * app->nchannels + 1 )*app->ntimes;

    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        app->theta[itheta] += stepsize * direction[itheta];
    }

    return 0;
}


template <typename myDouble> 
int
read_data(char *filename, myDouble *var, int size)
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
   printf("Reading file %s\n", filename);
   for ( i = 0; i < size; i++)
   {
      fscanf(file, "%lf", &tmp);
      var[i] = tmp;
   }

   /* close file */
   fclose(file);

   return 0;
}

template <typename myDouble>
int
write_data(char *filename, myDouble *var, int size)
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


/* Explicit instantiation of the template functions */
template int read_data<double>(char *filename, double *var, int size);
template int read_data<RealReverse>(char *filename, RealReverse *var, int size);

template int write_data<double>(char *filename, double *var, int size);
template int write_data<RealReverse>(char *filename, RealReverse *var, int size);

template double maximum<double>(double* a, int size_t);
template RealReverse maximum<RealReverse>(RealReverse* a, int size_t);

template double sigma<double>(double x);
template RealReverse sigma<RealReverse>(RealReverse x);

template int take_step<double>(double* Y, double* theta, int ts, double  dt, int *batch, int nbatch, int nchannels, int parabolic);
template int take_step<RealReverse>(RealReverse* Y, RealReverse* theta, int ts, double  dt, int *batch, int nbatch, int nchannels, int parabolic);

template double loss<double>(double  *Y, double *Target, int *batch, int nbatch, double *classW, double *classMu, int nclasses, int nchannels);
template RealReverse loss<RealReverse>(RealReverse *Y, double *Target, int *batch, int nbatch, RealReverse *classW, RealReverse *classMu, int nclasses, int nchannels);

template double regularization_theta<double>(double* theta, int ts, double dt, int ntime, int nchannels);
template RealReverse regularization_theta<RealReverse>(RealReverse* theta, int ts, double dt, int ntime, int nchannels);

template RealReverse regularization_class<RealReverse>(RealReverse *classW, RealReverse *classMu, int nclasses, int nchannels);
template double regularization_class<double>(double *classW, double *classMu, int nclasses, int nchannels);


    // /* --- CONSTRUCT A LABEL MATRIX --- */

    // double *Cstore = (double*) malloc(nclasses*nexamples*sizeof(double));
    // /* Read YTarget */
    // double  *Ytarget = (double*) malloc(nchannels * nexamples * sizeof(double));
    // read_data("Ytarget.transpose.dat", Ytarget, nchannels*nexamples);

    // /* multiply Ytarget with W and add mu */
    // int c_id, batch_id, weight_id, y_id;
    // for (int ibatch = 0; ibatch < nbatch; ibatch ++)
    // {
    //     batch_id = batch[ibatch];
    //     for (int iclass = 0; iclass < nclasses; iclass++)
    //     {
    //         c_id = batch_id * nclasses + iclass;
    //         Cstore[c_id] = 0.0;
        
    //         /* Apply classification weights */
    //         for (int ichannel = 0; ichannel < nchannels; ichannel++)
    //         {
    //             y_id          = batch_id * nchannels + ichannel;
    //             weight_id     = iclass   * nchannels + ichannel;
    //             Cstore[c_id] += Ytarget[y_id] * classW[weight_id];
    //         }

    //         /* Add classification bias */
    //         Cstore[c_id] += classMu[iclass];
    //     }
    // }

    // /* print Cstore to file */
    // write_data("Cstore.dat", Cstore, nclasses*nexamples);

    // free(Cstore);

    // /* Stop calculating */
    // return 0;

