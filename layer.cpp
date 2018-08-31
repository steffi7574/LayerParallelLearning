#include "layer.hpp"

Layer::Layer()
{
   dim_In       = 0;
   dim_Out      = 0;
   dim_Bias     = 0;
   ndesign      = 0;

   index        = 0;
   dt           = 0.0;
   weights      = NULL;
   weights_bar  = NULL;
   bias         = NULL;
   bias_bar     = NULL;
   activation   = NULL;
   dactivation  = NULL;
   update       = NULL;
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
   ndesign     = dimI * dimO + dimB;
   dt          = deltaT;
   activation  = Activ;
   dactivation = dActiv;
   
   update = new double[dimO];
}   

Layer::~Layer()
{
    delete update;
}


void Layer::setDt(double DT) { dt = DT; }

double* Layer::getWeights() { return weights; }
double* Layer::getBias()    { return bias; }

double* Layer::getWeightsBar() { return weights_bar; }
double* Layer::getBiasBar()    { return bias_bar; }

int Layer::getDimIn()   { return dim_In;   }
int Layer::getDimOut()  { return dim_Out;  }
int Layer::getDimBias() { return dim_Bias; }
int Layer::getnDesign() { return ndesign; }

void Layer::print_data(double* data)
{
    printf("DATA: ");
    for (int io = 0; io < dim_Out; io++)
    {
        printf("%1.14e ", data[io]);
    }
    printf("\n");
}

void Layer::initialize(double* design_ptr,
                       double* gradient_ptr,
                       double  factor)
{
    /* Set primal and adjoint weights memory locations */
    weights     = design_ptr;
    weights_bar = gradient_ptr;
    
    /* Set primal and adjoint bias memory locations */
    int nweights = dim_Out * dim_In;
    bias         = design_ptr + nweights;    
    bias_bar     = gradient_ptr + nweights;

    /* Initialize */
    for (int i = 0; i < ndesign - dim_Bias; i++)
    {
        weights[i]     = factor * (double) rand() / ((double) RAND_MAX);
        weights_bar[i] = 0.0;
    }
    for (int i = 0; i < ndesign - nweights; i++)
    {
        bias[i]     = factor * (double) rand() / ((double) RAND_MAX);
        bias_bar[i] = 0.0;
    }
}                   

void Layer::resetBar()
{
    for (int i = 0; i < ndesign - dim_Bias; i++)
    {
        weights_bar[i] = 0.0;
    }
    for (int i = 0; i < ndesign - dim_In * dim_Out; i++)
    {
        bias_bar[i] = 0.0;
    }
}


double Layer::evalTikh()
{
    double tik = 0.0;
    for (int i = 0; i < ndesign - dim_Bias; i++)
    {
        tik += pow(weights[i],2);
    }
    for (int i = 0; i < ndesign - dim_In * dim_Out; i++)
    {
        tik += pow(bias[i],2);
    }

    return tik / 2.0;
}

void Layer::evalTikh_diff(double regul_bar)
{
    /* Derivative bias term */
    for (int i = 0; i < ndesign - dim_In * dim_Out; i++)
    {
        bias_bar[i] += bias[i] * regul_bar;
    }
    for (int i = 0; i < ndesign - dim_Bias; i++)
    {
        weights_bar[i] += weights[i] * regul_bar;
    }
}

void Layer::setExample(double* example_ptr) {}

double Layer::evalLoss(double *data_Out, 
                       double *label) { return 0.0; }


void Layer::evalLoss_diff(double *data_Out, 
                          double *data_Out_bar,
                          double *label,
                          double  loss_bar) {}

int Layer::prediction(double* data) {return -1;}


DenseLayer::DenseLayer(int     idx,
                       int     dimI,
                       int     dimO,
                       double  deltaT,
                       double (*Activ)(double x),
                       double  (*dActiv)(double x)) : Layer(idx, dimI, dimO, 1, deltaT, Activ, dActiv)
{}
   
DenseLayer::~DenseLayer() {}


void DenseLayer::applyFWD(double* state)
{
   /* Compute update for each channel */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Apply weights */
      update[io] = vecdot(dim_In, &(weights[io*dim_In]), state);

      /* Add bias */
      update[io] += bias[0];

      /* apply activation */
      update[io] = activation(update[io]);
   }

      /* Apply step */
   for (int io = 0; io < dim_Out; io++)
   {
      state[io] = state[io] + dt * update[io];
   }
}


void DenseLayer::applyBWD(double* state,
                          double* state_bar)
{
   double tmp;

   /* Derivative of the update */
   for (int io = 0; io < dim_Out; io++)
   {
        update[io] = dt * state_bar[io];
   }

   for (int io = 0; io < dim_Out; io++)
   {
      /* Recompute intermediate state at actiation evaluation point */
      tmp  = vecdot(dim_In, &(weights[io*dim_In]), state);
      tmp += bias[0];

      /* Derivative of activation function */
      update[io] = dactivation(tmp) * update[io];

      /* Derivative of bias addition */
      bias_bar[0] += update[io];

      /* Derivative of weight application */
      for (int ii = 0; ii < dim_In; ii++)
      {
         weights_bar[io*dim_In + ii] += state[ii] * update[io];
         state_bar[ii] += weights[io*dim_In + ii] * update[io]; 
      }
   }
}


OpenDenseLayer::OpenDenseLayer(int     idx,
                               int     dimI,
                               int     dimO,
                               double  deltaT,
                               double (*Activ)(double x),
                               double (*dActiv)(double x)) : DenseLayer(idx,     dimI, dimO, deltaT, Activ, dActiv) 
{
    example = NULL;
}

OpenDenseLayer::~OpenDenseLayer(){}

void OpenDenseLayer::setExample(double* example_ptr)
{
    example = example_ptr;
}

void OpenDenseLayer::applyFWD(double* state) 
{
       /* Compute update for each channel */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Apply weights */
      state[io] = vecdot(dim_In, &(weights[io*dim_In]), example);

      /* Add bias */
      state[io] += bias[0];

      /* apply activation */
      state[io] = activation(state[io]);
   }
}

void OpenDenseLayer::applyBWD(double* state,
                              double* state_bar)
{
    double tmp;

   /* Backward propagation for each channel */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Recompute data_Out */
      tmp  = vecdot(dim_In, &(weights[io*dim_In]), state);
      tmp += bias[0];

      /* Derivative of activation function */
      state_bar[io] = dactivation(tmp) * state_bar[io];

      /* Derivative of bias addition */
      bias_bar[0] += state_bar[io];

      /* Derivative of weight application */
      for (int ii = 0; ii < dim_In; ii++)
      {
         weights_bar[io*dim_In + ii] += example[ii] * state_bar[io];
      }

      /* Reset */
      state_bar[io] = 0.0;
   }
}                


OpenExpandZero::OpenExpandZero(int    dimI,
                               int    dimO,
                               double deltaT) : Layer(0, dimI, dimO, 1, deltaT, NULL, NULL)
{
    /* this layer doesn't have any design variables. */ 
    ndesign = 0;
}


OpenExpandZero::~OpenExpandZero(){}


void OpenExpandZero::setExample(double* example_ptr)
{
    example = example_ptr;
}


void OpenExpandZero::applyFWD(double* state)
{
   for (int ii = 0; ii < dim_In; ii++)
   {
      state[ii] = example[ii];
   }
   for (int io = dim_In; io < dim_Out; io++)
   {
      state[io] = 0.0;
   }
}                           

void OpenExpandZero::applyBWD(double* state,
                              double* state_bar)
{
   for (int ii = 0; ii < dim_Out; ii++)
   {
      state_bar[ii] = 0.0;
   }
}                           


ClassificationLayer::ClassificationLayer(int idx,
                                         int dimI,
                                         int dimO,
                                         double  deltaT) : Layer(idx, dimI, dimO, dimO, deltaT, NULL, NULL)
{
    /* Allocate the probability vector */
    probability = new double[dimO];
}

ClassificationLayer::~ClassificationLayer()
{
    delete [] probability;
}


void ClassificationLayer::applyFWD(double* state)
{
    /* Compute update for each channel */
    for (int io = 0; io < dim_Out; io++)
    {
        /* Apply weights */
        update[io] = vecdot(dim_In, &(weights[io*dim_In]), state);
  
        /* Add bias */
        update[io] += bias[io];
    }

    /* Data normalization y - max(y) (needed for stable softmax evaluation */
    normalize(update);

    if (dim_In < dim_Out)
    {
        printf("Error: dimI < dimO on classification layer. Change implementation here. \n");
        exit(1);
    }
    for (int io = 0; io < dim_Out; io++)
    {
        state[io] = update[io];
    }
    /* Set remaining to zero */
    for (int ii = dim_Out; ii < dim_In; ii++)
    {
        state[ii] = 0.0;
    }
}                           
      
void ClassificationLayer::applyBWD(double* state,
                                   double* state_bar)
{
    double* tmpbar = new double[dim_Out];

    for (int ii = dim_Out; ii < dim_In; ii++)
    {
        state_bar[ii] = 0.0;
    }
    for (int io = 0; io < dim_Out; io++)
    {
        tmpbar[io]    = state_bar[io];
        state_bar[io] = 0.0;
    }
    
    /* Recompute data_out */
    for (int io = 0; io < dim_Out; io++)
    {
        update[io] = vecdot(dim_In, &(weights[io*dim_In]), state);
        update[io] += bias[io];
    }        

 
    /* Derivative of the normalization */
    normalize_diff(update, tmpbar);

    /* Backward propagation for each channel */
    for (int io = 0; io < dim_Out; io++)
    {
       /* Derivative of bias addition */
        bias_bar[io] += tmpbar[io];
  
        /* Derivative of weight application */
        for (int ii = 0; ii < dim_In; ii++)
        {
           weights_bar[io*dim_In + ii] += state[ii] * tmpbar[io];
           state_bar[ii] += weights[io*dim_In + ii] * tmpbar[io];
        }
    }   
    delete tmpbar;
}


void ClassificationLayer::normalize(double* data)
{

   /* Find maximum value */
   double max = vecmax(dim_Out, data);
   /* Shift the data vector */
   for (int io = 0; io < dim_Out; io++)
   {
       data[io] = data[io] - max;
   }
}   

void ClassificationLayer::normalize_diff(double* data, 
                                         double* data_bar)
{
    double max_b = 0.0;
    /* Derivative of the shift */
    for (int io = 0; io < dim_Out; io++)
    {
        max_b -= data_bar[io];
    }
    /* Derivative of the vecmax */
    int i_max = argvecmax(dim_Out, data);
    data_bar[i_max] += max_b;
}                                     

double ClassificationLayer::evalLoss(double *data_Out, 
                                      double *label) 
{
   double label_pr, exp_sum;
   double CELoss;

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
      
      
void ClassificationLayer::evalLoss_diff(double *data_Out, 
                                        double *data_Out_bar,
                                        double *label,
                                        double  loss_bar)
{
    double exp_sum, exp_sum_bar;
    double label_pr_bar = - loss_bar;

    /* Recompute exp_sum */
    exp_sum = 0.0;
    for (int io = 0; io < dim_Out; io++)
    {
       exp_sum += exp(data_Out[io]);
    }

    /* derivative of log(exp_sum) */
    exp_sum_bar  = 1./exp_sum * loss_bar;
    for (int io = 0; io < dim_Out; io++)
    {
        data_Out_bar[io] = exp(data_Out[io]) * exp_sum_bar;
    }

    /* Derivative of vecdot */
    for (int io = 0; io < dim_Out; io++)
    {
        data_Out_bar[io] +=  label[io] * label_pr_bar;
    }
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
       probability[io] = exp(data_Out[io]) / exp_sum;

      /* Predicted class is the one with maximum probability */ 
      if (probability[io] > max)
      {
          max      = probability[io]; 
          class_id = io; 
      }
   }

   return class_id;
}