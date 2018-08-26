#include "lib.hpp"


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


double ReLu_act(double x)
{
    return max(0.0, x); 
}

double d_ReLu_act(double x)
{
    double diff;
    if (x > 0.0) diff = 1.0;
    else         diff = 0.0;

    return diff;
}

double tanh_act(double x)
{
    return tanh(x);
}

double d_tanh_act(double x)
{
    double diff = 1.0 - pow(tanh(x),2);

    return diff;
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
               relax += pow( (theta[idx1] - theta[idx]) / dt, 2);
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
update_design(int       N, 
              double    stepsize,
              double   *direction,
              double   *design)
{
    for (int i = 0; i < N; i++)
    {
        design[i] -= stepsize * direction[i];
    }

    return 0;
}


double 
getWolfe(int     N,
         double* gradient,
         double* descentdir)
{
    /* compute the wolfe condition product */
    double wolfe = 0.0;
    for (int i = 0; i < N; i++)
    {
        wolfe += gradient[i] * descentdir[i];
    }

    return wolfe;
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
vector_normsq(int    size_t,
              double *vector)
{
    double norm = 0.0;
    for (int i = 0; i<size_t; i++)
    {
        norm += pow(getValue(vector[i]),2);
    }

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

void read_data(char    *filename, 
               double **var, 
               int      dimx, 
               int      dimy)
{
   FILE   *file;
   double  tmp;

   var = new double* [dimx];
   for (int ix = 0; ix<dimx; ix++)
   {
       var[ix] = new double[dimy];
   }

   /* Read data */
   file = fopen(filename, "r");
   if (file == NULL)
   {
      printf("Can't open %s \n", filename);
      exit(1);
   }
   printf("Reading file %s\n", filename);
   for (int ix = 0; ix < dimx; ix++)
   {
       for (int iy = 0; iy < dimy; iy++)
       {
            fscanf(file, "%lf", &tmp);
            var[ix][iy] = tmp;
       }
   }

   /* close file */
   fclose(file);
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
template int write_data<double>(char *filename, double *var, int size);
template int write_data<RealReverse>(char *filename, RealReverse *var, int size);

template double maximum<double>(double* a, int size_t);
template RealReverse maximum<RealReverse>(RealReverse* a, int size_t);

template double loss<double>(double  *Y, double *Target, double  *Ydata, double *classW, double *classMu, int nelem, int nclasses, int nchannels, int nfeatures, int output, int* success_ptr);
template RealReverse loss<RealReverse>(RealReverse *Y, double *Target, double *Ydata, RealReverse *classW, RealReverse *classMu, int nelem, int nclasses, int nchannels, int nfeatures, int output, int* success_ptr);

template double ddt_theta_regul<double>(double* theta, int ts, double dt, int ntime, int nchannels);
template RealReverse ddt_theta_regul<RealReverse>(RealReverse* theta, int ts, double dt, int ntime, int nchannels);

template double tikhonov_regul<double>(double *variable, int size);
template RealReverse tikhonov_regul<RealReverse>(RealReverse *variable, int size);

