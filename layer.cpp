#include "layer.hpp"

Layer::Layer()
{
   dim_In       = 0;
   dim_Out      = 0;
   dim_Bias     = 0;

   index        = 0;
   dt           = 0.0;
   weights      = NULL;
   weights_bar  = NULL;
   bias         = NULL;
   bias_bar     = NULL;
   activation   = NULL;
   dactivation  = NULL;
}

Layer::Layer(int     idx,
             int     dimI,
             int     dimO,
             int     dimB,
             double  deltaT,
             double (*Activ)(double x),
             double  (*dActiv)(double x))
{
   index       = idx;
   dim_In      = dimI;
   dim_Out     = dimO;
   dim_Bias    = dimB;
   dt          = deltaT;
   activation  = Activ;
   dactivation = dActiv;

   weights     = new double [dim_Out * dim_In];
   weights_bar = new double [dim_Out * dim_In];
   bias        = new double [dim_Bias];
   bias_bar    = new double [dim_Bias];
}   

Layer::~Layer()
{
   delete [] weights;
   delete [] weights_bar;
   delete [] bias;
   delete [] bias_bar;
}


void Layer::setDt(double DT)
{
   dt = DT;
}


void Layer::initialize(double factor)
{
    for (int i = 0; i < dim_Out * dim_In; i++)
    {
        weights[i]     = factor * (double) rand() / ((double) RAND_MAX);
        weights_bar[i] = 0.0;
    }
    for (int i = 0; i < dim_Bias; i++)
    {
        bias[i]     = factor * (double) rand() / ((double) RAND_MAX);
        bias_bar[i] = 0.0;
    }
}


DenseLayer::DenseLayer(int     idx,
                       int     dimI,
                       int     dimO,
                       double  deltaT,
                       double (*Activ)(double x),
                       double  (*dActiv)(double x)) : Layer(idx, dimI, dimO, 1, deltaT, Activ, dActiv)
{}
   
DenseLayer::~DenseLayer() {}


void DenseLayer::applyFWD(double* data_In,
                          double* data_Out)
{
   /* Compute update for each channel */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Apply weights */
      data_Out[io] = vecdot(dim_In, &(weights[io*dim_In]), data_In);

      /* Add bias */
      data_Out[io] += bias[0];

      /* apply activation */
      data_Out[io] = activation(data_Out[io]);

      /* Apply step */
      data_Out[io] = dt * data_Out[io];

      /* If not first layer, add the incoming data. */
      if (index != 0)
      {
         data_Out[io] = data_In[io];
      }
   }

}


void DenseLayer::applyBWD(double* data_In,
                          double* data_Out,
                          double* data_In_bar,
                          double* data_Out_bar)
{

   /* Backward propagation for each channel */
   for (int io = 0; io < dim_Out; io++)
   {
     
      /* Derivative of the update */
      if (index != 0)
      {
         data_In_bar[io] += data_Out_bar[io];
      }
      data_Out_bar[io] = dt*data_Out_bar[io];


      /* Recompute data_Out */
      data_Out[io] = vecdot(dim_In, &(weights[io*dim_In]), data_In);
      data_Out[io] += bias[0];

      /* Derivative of activation function */
      data_Out_bar[io] = dactivation(data_Out[io]) * data_Out_bar[io];

      /* Derivative of bias addition */
      bias_bar[0] += data_Out_bar[io];

      /* Derivative of weight application */
      for (int jo = 0; jo < dim_In; jo++)
      {
         data_In_bar[jo] += weights[io*dim_In + jo] * data_Out_bar[io];
         weights_bar[io*dim_In + jo] += data_In[jo] * data_Out_bar[io];
      }

      /* Reset */
      data_Out_bar[io] = 0.0;
   }
}


OpenExpandZero::OpenExpandZero(int  dimI,
                               int  dimO,
                               double  deltaT) : Layer(0, dimI, dimO, 1, deltaT, NULL, NULL){}


OpenExpandZero::~OpenExpandZero(){}


void OpenExpandZero::applyFWD(double* data_In, 
                              double* data_Out)
{
   for (int ii = 0; ii < dim_In; ii++)
   {
      data_Out[ii] = data_In[ii];
   }
   for (int io = dim_In; io < dim_Out; io++)
   {
      data_Out[io] = 0.0;
   }
}                           

void OpenExpandZero::applyBWD(double* data_In,
                              double* data_Out,
                              double* data_In_bar,
                              double* data_Out_bar)
{
   for (int ii = 0; ii < dim_In; ii++)
   {
      data_In_bar[ii] += data_Out_bar[ii];
      data_Out_bar[ii] = 0.0;
   }
}                           





ClassificationLayer::ClassificationLayer(int idx,
                                         int dimI,
                                         int dimO,
                                         double  deltaT) : Layer(idx, dimI, dimO, dimO, deltaT, NULL, NULL){}

ClassificationLayer::~ClassificationLayer(){}


void ClassificationLayer::applyFWD(double* data_In, 
                                   double* data_Out)
{
    /* Compute update for each channel */
    for (int io = 0; io < dim_Out; io++)
    {
        /* Apply weights */
        data_Out[io] = vecdot(dim_In, &(weights[io*dim_In]), data_In);
  
        /* Add bias */
        data_Out[io] += bias[io];
    }
}                           
      
void ClassificationLayer::applyBWD(double* data_In,
                                   double* data_Out,
                                   double* data_In_bar,
                                   double* data_Out_bar)
{
    /* Backward propagation for each channel */
    for (int io = 0; io < dim_Out; io++)
    {
       /* Derivative of bias addition */
        bias_bar[io] += data_Out_bar[io];
  
        /* Derivative of weight application */
        for (int ji = 0; ji < dim_In; ji++)
        {
           data_In_bar[ji] += weights[io*dim_In + ji] * data_Out_bar[io];
           weights_bar[io*dim_In + ji] += data_In[ji] * data_Out_bar[io];
        }
  
        /* Reset */
        data_Out_bar[io] = 0.0;
    }   
}           