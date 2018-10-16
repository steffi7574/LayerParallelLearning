#include "layer.hpp"

Layer::Layer()
{
   dim_In       = 0;
   dim_Out      = 0;
   dim_Bias     = 0;
   ndesign      = 0;

   index        = 0;
   dt           = 0.0;
   activ        = -1;
   weights      = NULL;
   weights_bar  = NULL;
   bias         = NULL;
   bias_bar     = NULL;
   gamma        = 0.0;
   update       = NULL;
   update_bar   = NULL;
}

Layer::Layer(int     idx,
             int     dimI,
             int     dimO,
             int     dimB,
             double  deltaT,
             int     Activ,
             double  Gamma)
{
   index       = idx;
   dim_In      = dimI;
   dim_Out     = dimO;
   dim_Bias    = dimB;
   ndesign     = dimI * dimO + dimB;
   dt          = deltaT;
   activ       = Activ;
   gamma       = Gamma;
   
   update     = new double[dimO];
   update_bar = new double[dimO];
}   
 
// Layer::Layer(0, dimI, dimO, 1)
Layer::Layer(int idx, 
             int dimI, 
             int dimO, 
             int dimB) : Layer(idx, dimI, dimO, dimB, 1.0, -1, 0.0) {}         

Layer::~Layer()
{
    delete [] update;
    delete [] update_bar;
}


void Layer::setDt(double DT) { dt = DT; }

double Layer::getDt() { return dt; }

double Layer::getGamma() { return gamma; }

int Layer::getActivation() { return activ; }

double* Layer::getWeights() { return weights; }
double* Layer::getBias()    { return bias; }

double* Layer::getWeightsBar() { return weights_bar; }
double* Layer::getBiasBar()    { return bias_bar; }

int Layer::getDimIn()   { return dim_In;   }
int Layer::getDimOut()  { return dim_Out;  }
int Layer::getDimBias() { return dim_Bias; }
int Layer::getnDesign() { return ndesign; }

int Layer::getIndex() { return index; }

void Layer::print_data(double* data)
{
    printf("DATA: ");
    for (int io = 0; io < dim_Out; io++)
    {
        printf("%1.14e ", data[io]);
    }
    printf("\n");
}


double Layer::activation(double x)
{
    double y;
    switch ( activ )
    {
       case TANH:
          y = Layer::tanh_act(x);
          break;
       case RELU:
          y = Layer::ReLu_act(x);
          break;
       case SMRELU:
          y = Layer::SmoothReLu_act(x);
          break;
       default:
          printf("ERROR: You should specify an activation function!\n");
          printf("GO HOME AND GET SOME SLEEP!");
    }
    return y;
}

double Layer::dactivation(double x)
{
    double y;
    switch ( activ)
    {
       case TANH:
          y = Layer::dtanh_act(x);
          break;
       case RELU:
          y = Layer::dReLu_act(x);
          break;
       case SMRELU:
          y = Layer::dSmoothReLu_act(x);
          break;
       default:
          printf("ERROR: You should specify an activation function!\n");
          printf("GO HOME AND GET SOME SLEEP!");
    }
    return y;

}

void Layer::initialize(double* design_ptr,
                       double* gradient_ptr,
                       double  factor)
{
    /* Set primal and adjoint weights memory locations */
    weights     = design_ptr;
    weights_bar = gradient_ptr;
    
    /* Bias memory locations is a shift by number of weights */
    int nweights = dim_Out * dim_In;
    bias         = design_ptr + nweights;    
    bias_bar     = gradient_ptr + nweights;

    /* Initialize */
    for (int i = 0; i < ndesign - dim_Bias; i++)
    {
        // weights[i]     = factor * (double) rand() / ((double) RAND_MAX);
        weights[i]     = factor * i * index ;
        weights_bar[i] = 0.0;
    }
    for (int i = 0; i < ndesign - nweights; i++)
    {
        // bias[i]     = factor * (double) rand() / ((double) RAND_MAX);
        bias[i]     = factor * i * index;
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

    return gamma / 2.0 * tik;
}

void Layer::evalTikh_diff(double regul_bar)
{
    regul_bar = gamma * regul_bar;

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
                          double  loss_bar) 
{
    /* dfdu = 0.0 */
    for (int io = 0; io < dim_Out; io++)
    {
        data_Out_bar[io] =  0.0;
    }
}

int Layer::prediction(double* data, double* label) {return 0;}


DenseLayer::DenseLayer(int     idx,
                       int     dimI,
                       int     dimO,
                       double  deltaT,
                       int     Activ,
                       double  Gamma) : Layer(idx, dimI, dimO, 1, deltaT, Activ, Gamma)
{}
   
DenseLayer::~DenseLayer() {}


void DenseLayer::applyFWD(double* state)
{
   /* Affine transformation */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Apply weights */
      update[io] = vecdot(dim_In, &(weights[io*dim_In]), state);

      /* Add bias */
      update[io] += bias[0];
   }

      /* Apply step */
   for (int io = 0; io < dim_Out; io++)
   {
      state[io] = state[io] + dt * activation(update[io]);
   }
}


void DenseLayer::applyBWD(double* state,
                          double* state_bar)
{

   /* Derivative of the step */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Recompute affine transformation */
        update[io]  = vecdot(dim_In, &(weights[io*dim_In]), state);
        update[io] += bias[0];
        
        /* Derivative */
        update_bar[io] = dt * dactivation(update[io]) * state_bar[io];
   }

    /* Derivative of linear transformation */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Derivative of bias addition */
      bias_bar[0] += update_bar[io];

      /* Derivative of weight application */
      for (int ii = 0; ii < dim_In; ii++)
      {
         weights_bar[io*dim_In + ii] += state[ii] * update_bar[io];
         state_bar[ii] += weights[io*dim_In + ii] * update_bar[io]; 
      }
   }
}


OpenDenseLayer::OpenDenseLayer(int     dimI,
                               int     dimO,
                               int     Activ,
                               double  Gamma) : DenseLayer(0, dimI, dimO, 1.0, Activ, Gamma) 
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
   /* affine transformation */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Apply weights */
      update[io] = vecdot(dim_In, &(weights[io*dim_In]), example);

      /* Add bias */
      update[io] += bias[0];
   }

   /* Step */
   for (int io = 0; io < dim_Out; io++)
   {
      state[io] = activation(update[io]);
   }
}

void OpenDenseLayer::applyBWD(double* state,
                              double* state_bar)
{
   /* Derivative of step */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Recompute affine transformation */
      update[io]  = vecdot(dim_In, &(weights[io*dim_In]), example);
      update[io] += bias[0];

      /* Derivative */
      update_bar[io] = dactivation(update[io]) * state_bar[io];
      state_bar[io] = 0.0;
   }

   /* Derivative of affine transformation */
   for (int io = 0; io < dim_Out; io++)
   {
      /* Derivative of bias addition */
      bias_bar[0] += update_bar[io];

      /* Derivative of weight application */
      for (int ii = 0; ii < dim_In; ii++)
      {
         weights_bar[io*dim_In + ii] += example[ii] * update_bar[io];
      }
   }
}                


OpenExpandZero::OpenExpandZero(int dimI,
                               int dimO) : Layer(0, dimI, dimO, 1)
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


ClassificationLayer::ClassificationLayer(int    idx,
                                         int    dimI,
                                         int    dimO,
                                         double Gamma) : Layer(idx, dimI, dimO, dimO)
{
    gamma = Gamma;
    /* Allocate the probability vector */
    probability = new double[dimO];
}

ClassificationLayer::~ClassificationLayer()
{
    delete [] probability;
}


void ClassificationLayer::applyFWD(double* state)
{
    /* Compute affine transformation */
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
        printf("Error: nchannels < nclasses. Implementation of classification layer doesn't support this setting. Change! \n");
        exit(1);
    }

    /* Apply step */
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
    /* Recompute affine transformation */
    for (int io = 0; io < dim_Out; io++)
    {
        update[io] = vecdot(dim_In, &(weights[io*dim_In]), state);
        update[io] += bias[io];
    }        


    /* Derivative of step */
    for (int ii = dim_Out; ii < dim_In; ii++)
    {
        state_bar[ii] = 0.0;
    }
    for (int io = 0; io < dim_Out; io++)
    {
        update_bar[io] = state_bar[io];
        state_bar[io]  = 0.0;
    }
    
    /* Derivative of the normalization */
    normalize_diff(update, update_bar);

    /* Derivatie of affine transformation */
    for (int io = 0; io < dim_Out; io++)
    {
       /* Derivative of bias addition */
        bias_bar[io] += update_bar[io];
  
        /* Derivative of weight application */
        for (int ii = 0; ii < dim_In; ii++)
        {
           weights_bar[io*dim_In + ii] += state[ii] * update_bar[io];
           state_bar[ii] += weights[io*dim_In + ii] * update_bar[io];
        }
    }   
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


int ClassificationLayer::prediction(double* data_Out, 
                                    double* label)
{
   double exp_sum, max;
   int    class_id;
   int    success = 0;

   /* Compute sum_i (exp(x_i)) */
   max = -1.0;
   exp_sum = 0.0;
   for (int io = 0; io < dim_Out; io++)
   {
      exp_sum += exp(data_Out[io]);
   }

   for (int io = 0; io < dim_Out; io++)
   {
       /* Compute class probabilities (Softmax) */
       probability[io] = exp(data_Out[io]) / exp_sum;

      /* Predicted class is the one with maximum probability */ 
      if (probability[io] > max)
      {
          max      = probability[io]; 
          class_id = io; 
      }
   }

  /* Test for successful prediction */
  if ( label[class_id] > 0.99 )  
  {
      success = 1;
  }
   

   return success;
}



double Layer::ReLu_act(double x)
{
    return std::max(0.0, x);
}


double Layer::dReLu_act(double x)
{
    double diff;
    if (x >= 0.0) diff = 1.0;
    else         diff = 0.0;

    return diff;
}


double Layer::SmoothReLu_act(double x)
{
    /* range of quadratic interpolation */
    double eta = 0.1;
    /* Coefficients of quadratic interpolation */
    double a   = 1./(4.*eta);
    double b   = 1./2.;
    double c   = eta / 4.;

    if (-eta < x && x < eta)
    {
        /* Quadratic Activation */
        return a*pow(x,2) + b*x + c;
    }
    else
    {
        /* ReLu Activation */
        return Layer::ReLu_act(x);
    }
}

double Layer::dSmoothReLu_act(double x)
{
    /* range of quadratic interpolation */
    double eta = 0.1;
    /* Coefficients of quadratic interpolation */
    double a   = 1./(4.*eta);
    double b   = 1./2.;

    if (-eta < x && x < eta)
    {
        return 2.*a*x + b;
    }
    else
    {
        return Layer::dReLu_act(x);
    }

}


double Layer::tanh_act(double x)
{
    return tanh(x);
}

double Layer::dtanh_act(double x)
{
    double diff = 1.0 - pow(tanh(x),2);

    return diff;
}