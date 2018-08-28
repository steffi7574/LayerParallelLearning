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


void Layer::setDt(double DT) { dt = DT; }

double* Layer::getWeights() { return weights; }
double* Layer::getBias()    { return bias; }

int Layer::getDimIn()   { return dim_In;   }
int Layer::getDimOut()  { return dim_Out;  }
int Layer::getDimBias() { return dim_Bias; }


void Layer::print_data(double* data)
{
    printf("DATA: ");
    for (int io = 0; io < dim_Out; io++)
    {
        printf("%1.14e ", data[io]);
    }
    printf("\n");
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

double Layer::evalTikh()
{
    double tik = 0.0;
    for (int i = 0; i < dim_Out * dim_In; i++)
    {
        tik += pow(weights[i],2);
    }
    for (int i = 0; i < dim_Bias; i++)
    {
        tik += pow(bias[i],2);
    }

    return tik / 2.0;
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
         if (index != 0) data_In_bar[jo] += weights[io*dim_In + jo] * data_Out_bar[io]; // at first layer: data_in is the input data, its derivative is not needed. 
         weights_bar[io*dim_In + jo] += data_In[jo] * data_Out_bar[io];
      }

      /* Reset */
      data_Out_bar[io] = 0.0;
   }
}

double DenseLayer::evalLoss(double *data_Out, double *label) { return 0.0; }
int    DenseLayer::prediction(double* data) {return -1;}


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


double OpenExpandZero::evalLoss(double *data_Out, double *label){ return 0.0;}
int    OpenExpandZero::prediction(double* data) {return -1;}


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


void ClassificationLayer::normalize(double* data)
{

   double max = vecmax(dim_Out, data);
   for (int io = 0; io < dim_Out; io++)
   {
       data[io] = data[io] - max;
   }

}   

double ClassificationLayer::evalLoss(double *data_Out, 
                                      double *label) 
{
   double label_pr, exp_sum;
   double CELoss;

   /* Data normalization y - max(y) */
   normalize(data_Out);

   /* Label projection */
   label_pr = vecdot(dim_Out, label, data_Out);

   /* Compute sum_i (exp(x_i)) */
   exp_sum = 0.0;
   for (int io = 0; io < dim_Out; io++)
   {
      exp_sum += exp(data_Out[io]);
   }

   /* Cross entropy loss function */
   CELoss = - label_pr + log(exp_sum);

   return CELoss;
}


int ClassificationLayer::prediction(double* data_Out)
{
   double exp_sum, max;
   int    class_id;

   /* Compute sum_i (exp(x_i)) */
   max = -1.0;
   exp_sum = 0.0;
   for (int io = 0; io < dim_Out; io++)
   {
      exp_sum += exp(data_Out[io]);
   }

   /* Compute class probabilities (Softmax) */
   for (int io = 0; io < dim_Out; io++)
   {
       data_Out[io] = exp(data_Out[io]) / exp_sum;

      /* Predicted class is the one with maximum probability */ 
      if (data_Out[io] > max)
      {
          max      = data_Out[io]; 
          class_id = io; 
      }
   }

   return class_id;
}