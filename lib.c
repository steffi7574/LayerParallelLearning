#include "lib.h"


template <typename myDouble>
myDouble 
maximum(myDouble *a,
        int       size_t)
{
   myDouble max = - 1e+12;
   
   for (int i = 0; i < size_t; i++)
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


int
opening_expand(double *Y, 
               double *data, 
               int     nelem, 
               int     nchannels, 
               int     nfeatures)
{
    int idata = 0;
    for (int ielem = 0; ielem < nelem; ielem++)
    {
        Y[ielem*nchannels + 0] = data[idata];
        idata++;
        Y[ielem*nchannels + 1] = data[idata];
        idata++;
        for (int ichannels = 2; ichannels < nchannels; ichannels++)
        {
            Y[ielem*nchannels + ichannels] = 0.0;
        }
    }

    return 0;
}           


template <typename myDouble>
int
opening_layer(myDouble *Y,
              myDouble *theta_open, 
              double   *Ydata, 
              int nelem, 
              int nchannels, 
              int nfeatures)
{
    myDouble sum;
    int      y_id, k_id, bias_id;

    for (int ielem = 0; ielem < nelem; ielem++)
    {
        for (int ichannels = 0; ichannels < nchannels; ichannels++)
        {
            /* Apply K matrix and bias */
            sum = 0.0;
            for (int ifeatures = 0; ifeatures < nfeatures; ifeatures++)
            {
                y_id = ielem * nfeatures + ifeatures;
                k_id = ifeatures * nchannels + ichannels;
                sum += Ydata[y_id] * theta_open[k_id];
            }
            bias_id = nfeatures * nchannels;
            sum += theta_open[bias_id];

            /* Apply nonlinear activation */
            y_id = ielem * nchannels + ichannels;
            Y[y_id] = sigma(sum);
        }
    }

    return 0;
}


template <typename myDouble>
int
take_step(myDouble* Y,
          myDouble* theta,
          int     ts,
          double  dt,
          int     nelem,
          int     nchannels, 
          int     parabolic)
{
   /* Element Y_id stored in Y[id * nf, ..., ,(id+1)*nf -1] */
   myDouble sum;
   int    th_idx;
   myDouble *update = (myDouble*)malloc(nchannels * sizeof(myDouble));

   /* iterate over all elements */ 
   for (int ielem = 0; ielem < nelem; ielem++)
   {
      /* Iterate over all channels */
      for (int ichannel = 0; ichannel < nchannels; ichannel++)
      {
         /* Apply weights */
         sum = 0.0;
         for (int jchannel = 0; jchannel < nchannels; jchannel++)
         {
            th_idx = ts * ( nchannels * nchannels + 1) + jchannel * nchannels + ichannel;
            sum += theta[th_idx] * Y[ielem * nchannels + jchannel];
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

     
         int idx = ielem * nchannels + ichannel;
         Y[idx] += dt * sum;

        //  printf("%d %d u->Y[idx] %1.14e\n", ielem, ichannel, getValue(Y[idx]));
      }
   }      

   free(update);
   return 0;
}


template <typename myDouble>
myDouble
loss(myDouble     *Y,
     double       *Target,
     double       *Ydata,
     myDouble     *classW,
     myDouble     *classMu,
     int           nelem,
     int           nclasses,
     int           nchannels,
     int           nfeatures,
     int           output,
     int          *success_ptr)
{
   myDouble loss_sum, loss_local, normalization, sum; 
   int      weight_id, y_id, target_id;
   int      predicted_class;
   int      success;
   double   max, probability;             
   FILE    *predictionfile;

   myDouble* YW_elem = new myDouble [nclasses];

   /* Open file for output of prediction data */
   if (output)
   {
       predictionfile = fopen("prediction.dat", "w");
       fprintf(predictionfile, "# x-coord      y-coord     predicted class\n");
   }

   success = 0;
   loss_sum = 0.0;
   /* Loop over elements */
   for (int ielem = 0; ielem < nelem; ielem ++)
   {
        /* Apply the classification weights YW */
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            YW_elem[iclass] = 0.0;

            for (int ichannel = 0; ichannel < nchannels; ichannel++)
            {
                y_id      = ielem * nchannels + ichannel;
                weight_id = iclass   * nchannels + ichannel;
                YW_elem[iclass] += Y[y_id] * classW[weight_id];
            }
        }

        /* Add the classification bias YW + mu */
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            YW_elem[iclass] += classMu[iclass];
        }


        /* Pointwise Normalization YW + mu - max(classes) */
        normalization = maximum (YW_elem, nclasses);
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            YW_elem[iclass] -= normalization;
        }


        /* First cross entrpy term: sum (C.*YW) */
        sum = 0.0;
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            target_id = ielem * nclasses + iclass;
            sum += Target[target_id] * YW_elem[iclass];
        }
        loss_local = -sum;

        /* Second cross entropy term: log(sum(exp.(YW))) */
        sum = 0.0;
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            sum += exp(YW_elem[iclass]);
        }
        loss_local += log(sum);

        /* Add to loss */
        loss_sum += loss_local;

        /* Get the predicted class label (Softmax) */
        max             = -1.0;
        for (int iclass = 0; iclass < nclasses; iclass++)
        {
            probability = exp(getValue(YW_elem[iclass])) / getValue(sum);
            if (probability > max)
            {
                max             = probability; 
                predicted_class = iclass; 
            }
        }

        /* Test for success */
        target_id = ielem * nclasses + predicted_class;
        if (Target[target_id] > 0.5)
        {
            success++;
        }

        /* Print prediction to file */
        if (output)
        {
            for (int ifeatures = 0; ifeatures < nfeatures; ifeatures++)
            {
                y_id = ielem * nchannels + ifeatures;
                fprintf(predictionfile, "%1.8e  ", Ydata[y_id]);
            }
            fprintf(predictionfile, "%d\n", predicted_class);
        }
   }

   *success_ptr = success;  

   delete [] YW_elem;
   if (output)
   {
       fclose(predictionfile);
       printf("File written: prediction.dat\n");
   }

   return loss_sum;
}


template <typename myDouble>
myDouble
tikhonov_regul(myDouble *variable,
               int       size)
{
    myDouble tik = 0.0;
    for (int idx = 0; idx < size; idx++)
    {
        tik += pow(variable[idx],2);
    }

    return tik / 2.0;
}           


template <typename myDouble>
myDouble
ddt_theta_regul(myDouble* theta,
                int       ts,
                double    dt,
                int       ntime,
                int       nchannels)
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
  
            if (ts < ntime - 1) 
            {
               relax += pow( (theta[idx1] - theta[idx]) / dt,2 );
            } 
        }
    }

    /* b(theta)-part */
    idx  =   ts     * ( nchannels * nchannels + 1) + nchannels*nchannels;
    if (ts < ntime - 1)
    {
        idx1 = ( ts+1 ) * ( nchannels * nchannels + 1) + nchannels*nchannels;
        relax += pow( (theta[idx1] - theta[idx]) / dt, 2 );
    }

    return relax / 2.0;
}        


int
collect_gradient(braid_App    app, 
                   MPI_Comm   comm,
                   double    *gradient)
{

   int ntheta_open = app->nfeatures * app->nchannels + 1;
   int ntheta      = (app->nchannels * app->nchannels + 1) * app->ntimes;
   int nclassW     = app->nchannels * app->nclasses;
   int nclassmu    = app->nclasses;
   int igradient   = 0;

   double* local_grad = (double*) malloc((ntheta_open + ntheta + nclassW + nclassmu)*sizeof(double));

   for (int itheta_open = 0; itheta_open < ntheta_open; itheta_open++)
   {
       local_grad[igradient] = app->theta_open_grad[itheta_open];
       igradient++;
   }
   for (int itheta = 0; itheta < ntheta; itheta++)
   {
       local_grad[igradient] = app->theta_grad[itheta];
       igradient++;
   }
   for (int iclassW = 0; iclassW < nclassW; iclassW++)
   {
       local_grad[igradient] = app->classW_grad[iclassW];
       igradient++;
   }
   for (int iclassmu = 0; iclassmu < nclassmu; iclassmu++)
   {
       local_grad[igradient] = app->classMu_grad[iclassmu];
       igradient++;
   }

   /* Collect sensitivities from all time-processors */
   MPI_Allreduce(local_grad, gradient, igradient, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   free(local_grad);

   return 0;
}                  




// int
// gradient_norm(braid_App app,
//               double   *theta_gnorm_prt,
//               double   *class_gnorm_prt)

// {
//     double ntheta = (app->nchannels * app->nchannels + 1) * app->ntimes;
//     int nclassW   =  app->nchannels * app->nclasses;
//     int nclassmu  =  app->nclasses;
//     double theta_gnorm, class_gnorm;

//     /* Norm of gradient */
//     theta_gnorm = 0.0;
//     class_gnorm = 0.0;
//     for (int itheta = 0; itheta < ntheta; itheta++)
//     {
//         theta_gnorm += pow(getValue(app->theta_grad[itheta]), 2);
//     }
//     for (int i = 0; i < nclassW; i++)
//     {
//         class_gnorm += pow(getValue(app->classW_grad[i]),2);
//     }
//     for (int i = 0; i < nclassmu; i++)
//     {
//         class_gnorm += pow(getValue(app->classMu_grad[i]),2);
//     }
//     theta_gnorm = sqrt(theta_gnorm);
//     class_gnorm = sqrt(class_gnorm);

//     *theta_gnorm_prt = theta_gnorm;
//     *class_gnorm_prt = class_gnorm;
    
//     return 0;
// }
    
int 
update_design(int       N, 
              double    stepsize,
              double   *direction,
              double   *design)
{
    for (int i = 0; i < N; i++)
    {
        design[i] += stepsize * direction[i];
    }

    return 0;
}



double
compute_descentdir(int     N,
               double* Hessian,
               double* gradient,
               double* descentdir)
{
    double wolfe = 0.0;

    for (int i = 0; i < N; i++)
    {
        /* Compute the descent direction */
        descentdir[i] = 0.0;
        for (int j = 0; j < N; j++)
        {
            descentdir[i] -= Hessian[i*N + j] * gradient[j];
        }
        /* compute the wolfe condition product */
        wolfe += gradient[i] * descentdir[i];
    }

    return wolfe;
}        


int
copy_vector(int N, 
            double* u, 
            double* u_copy)
{
    for (int i=0; i<N; i++)
    {
        u_copy[i] = u[i];
    }

    return 0;
}


double
vector_norm(int    size_t,
            double *vector)
{
    double norm = 0.0;
    for (int i = 0; i<size_t; i++)
    {
        norm += pow(vector[i],2);
    }
    norm = sqrt(norm);

    return norm;
}


int
concat_4vectors(int     size1,
                double *vec1,
                int     size2,
                double *vec2,
                int     size3,
                double *vec3,
                int     size4,
                double *vec4,
                double *globalvec)
{
    int iglob = 0;
    for (int i = 0; i < size1; i++)
    {
        globalvec[iglob] = vec1[i];
        iglob++;
    }
    for (int i = 0; i < size2; i++)
    {
        globalvec[iglob] = vec2[i];
        iglob++;
    }
    for (int i = 0; i < size3; i++)
    {
        globalvec[iglob] = vec3[i];
        iglob++;
    }
    for (int i = 0; i < size4; i++)
    {
        globalvec[iglob] = vec4[i];
        iglob++;
    }
    return 0;
}

int
split_into_4vectors(double *globalvec,
                    int     size1,
                    double *vec1,
                    int     size2,
                    double *vec2,
                    int     size3,
                    double *vec3,
                    int     size4,
                    double *vec4)
{
    int iglob = 0;
    for (int i = 0; i < size1; i++)
    {
        vec1[i] = globalvec[iglob];
        iglob++;
    }
    for (int i = 0; i < size2; i++)
    {
        vec2[i] = globalvec[iglob];
        iglob++;
    }
    for (int i = 0; i < size3; i++)
    {
        vec3[i] = globalvec[iglob];
        iglob++;
    }
    for (int i = 0; i < size4; i++)
    {
        vec4[i] = globalvec[iglob];
        iglob++;
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

template int take_step<double>(double* Y, double* theta, int ts, double  dt, int nelem, int nchannels, int parabolic);
template int take_step<RealReverse>(RealReverse* Y, RealReverse* theta, int ts, double  dt, int nelem, int nchannels, int parabolic);

template double loss<double>(double  *Y, double *Target, double  *Ydata, double *classW, double *classMu, int nelem, int nclasses, int nchannels, int nfeatures, int output, int* success_ptr);
template RealReverse loss<RealReverse>(RealReverse *Y, double *Target, double *Ydata, RealReverse *classW, RealReverse *classMu, int nelem, int nclasses, int nchannels, int nfeatures, int output, int* success_ptr);

template double ddt_theta_regul<double>(double* theta, int ts, double dt, int ntime, int nchannels);
template RealReverse ddt_theta_regul<RealReverse>(RealReverse* theta, int ts, double dt, int ntime, int nchannels);

template double tikhonov_regul<double>(double *variable, int size);
template RealReverse tikhonov_regul<RealReverse>(RealReverse *variable, int size);

template int opening_layer<RealReverse>(RealReverse *Y, RealReverse *theta_open, double *Ydata, int nelem, int nchannels, int nfeatures);
template int opening_layer<double>(double *Y, double *theta_open, double *Ydata, int nelem, int nchannels, int nfeatures);