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
   gamma_tik    = 0.0;
   gamma_ddt    = 0.0;
   update       = NULL;
   update_bar   = NULL;
}

Layer::Layer(int     idx,
             int     Type,
             int     dimI,
             int     dimO,
             int     dimB,
             double  deltaT,
             int     Activ,
             double  gammatik,
             double  gammaddt)
{
   index       = idx;
   type        = Type;
   dim_In      = dimI;
   dim_Out     = dimO;
   dim_Bias    = dimB;
   ndesign     = dimI * dimO + dimB;
   dt          = deltaT;
   activ       = Activ;
   gamma_tik   = gammatik;
   gamma_ddt   = gammaddt;
   
   update     = new double[dimO];
   update_bar = new double[dimO];
}   
 
// Layer::Layer(0, dimI, dimO, 1)
Layer::Layer(int idx, 
             int type,
             int dimI, 
             int dimO, 
             int dimB) : Layer(idx, type, dimI, dimO, dimB, 1.0, -1, 0.0, 0.0) {}

Layer::~Layer()
{
    delete [] update;
    delete [] update_bar;
}


void Layer::setDt(double DT) { dt = DT; }

double Layer::getDt() { return dt; }

double Layer::getGammaTik() { return gamma_tik; }

double Layer::getGammaDDT() { return gamma_ddt; }

int Layer::getActivation() { return activ; }

int Layer::getType() { return type; }

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


void Layer::packDesign(double* buffer, 
                       int     size)
{
    int nweights = getnDesign() - getDimBias();
    int nbias    = getnDesign() - getDimIn() * getDimOut();
    int idx = 0;
    for (int i = 0; i < nweights; i++)
    {
        buffer[idx] = getWeights()[i];   idx++;
    }
    for (int i = 0; i < nbias; i++)
    {
        buffer[idx] = getBias()[i];     idx++;
    }
    /* Set the rest to zero */
    for (int i = idx; i < size; i++)
    {
        buffer[idx] = 0.0;  idx++;
    }
}

void Layer::unpackDesign(double* buffer)
{
    int nweights     = getnDesign() - getDimBias();
    int nbias        = getnDesign() - getDimIn() * getDimOut();

    int idx = 0;
    for (int i = 0; i < nweights; i++)
    {
        getWeights()[i] = buffer[idx]; idx++;
    }
    for (int i = 0; i < nbias; i++)
    {
        getBias()[i] = buffer[idx];   idx++;
    }
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
        weights[i]     = factor * (double) rand() / ((double) RAND_MAX);
        // weights[i]     = factor * i * index ;
        weights_bar[i] = 0.0;
    }
    for (int i = 0; i < ndesign - nweights; i++)
    {
        bias[i]     = factor * (double) rand() / ((double) RAND_MAX);
        // bias[i]     = factor * i * index;
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

    return gamma_tik / 2.0 * tik;
}

void Layer::evalTikh_diff(double regul_bar)
{
    regul_bar = gamma_tik * regul_bar;

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


double Layer::evalRegulDDT(Layer* layer_prev, 
                           double deltat)
{
    if (layer_prev == NULL) return 0.0;

    double diff;
    double regul_ddt = 0.0;

    /* Compute ddt-regularization only if dimensions match  */
    /* this excludes openinglayer, first layer and classification layer. */
    if (layer_prev->getnDesign() == ndesign   &&
        layer_prev->getDimIn()   == dim_In    &&
        layer_prev->getDimOut()  == dim_Out   &&
        layer_prev->getDimBias() == dim_Bias   )
    {
        int nweights = getnDesign() - getDimBias();
        for (int iw = 0; iw < nweights; iw++)
        {
            diff = (getWeights()[iw] - layer_prev->getWeights()[iw]) / deltat;
            regul_ddt += pow(diff,2);
        }
        int nbias = getnDesign() - getDimIn() * getDimOut();
        for (int ib = 0; ib < nbias; ib++)
        {
            diff       = (getBias()[ib] - layer_prev->getBias()[ib]) / deltat;
            regul_ddt += pow(diff,2);
        }
        regul_ddt = gamma_ddt / 2.0 * regul_ddt;
    }

    return regul_ddt;
}                

void Layer::evalRegulDDT_diff(Layer* layer_prev, 
                              Layer* layer_next,
                              double deltat)
{

    if (layer_prev == NULL) return;
    if (layer_next == NULL) return;

    double diff;
    int regul_bar = gamma_ddt / (deltat*deltat);

    /* Left sided derivative term */
    if (layer_prev->getnDesign() == ndesign   &&
        layer_prev->getDimIn()   == dim_In    &&
        layer_prev->getDimOut()  == dim_Out   &&
        layer_prev->getDimBias() == dim_Bias   )
    {
        int nbias = getnDesign() - getDimIn() * getDimOut();
        for (int ib = 0; ib < nbias; ib++)
        {
            diff              = getBias()[ib] - layer_prev->getBias()[ib];
            getBiasBar()[ib] += diff * regul_bar;
        }

        int nweights = getnDesign() - getDimBias();
        for (int iw = 0; iw < nweights; iw++)
        {
            diff                 = getWeights()[iw] - layer_prev->getWeights()[iw];
            getWeightsBar()[iw] += diff * regul_bar;
        }
    }

    /* Right sided derivative term */
    if (layer_next->getnDesign() == ndesign   &&
        layer_next->getDimIn()   == dim_In    &&
        layer_next->getDimOut()  == dim_Out   &&
        layer_next->getDimBias() == dim_Bias   )
    {
        int nbias = getnDesign() - getDimIn() * getDimOut();
        for (int ib = 0; ib < nbias; ib++)
        {
            diff              = getBias()[ib] - layer_next->getBias()[ib];
            getBiasBar()[ib] += diff * regul_bar;
        }

        int nweights = getnDesign() - getDimBias();
        for (int iw = 0; iw < nweights; iw++)
        {
            diff                 = getWeights()[iw] - layer_next->getWeights()[iw];
            getWeightsBar()[iw] += diff * regul_bar;
        }
    }
} 




void Layer::setExample(double* example_ptr) {}


void Layer::evalClassification(int      nexamples, 
                               double** state,
                               double** labels, 
                               double*  loss_ptr, 
                               double*  accuracy_ptr)
{
    *loss_ptr     = 0.0;
    *accuracy_ptr = 0.0;
}


void Layer::evalClassification_diff(int      nexamples, 
                                    double** primalstate,
                                    double** adjointstate,
                                    double** labels, 
                                    int      compute_gradient) {}                                


DenseLayer::DenseLayer(int     idx,
                       int     dimI,
                       int     dimO,
                       double  deltaT,
                       int     Activ,
                       double  gammatik, 
                       double  gammaddt) : Layer(idx, DENSE, dimI, dimO, 1, deltaT, Activ, gammatik, gammaddt)
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
                          double* state_bar,
                          int     compute_gradient)
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
      if (compute_gradient) bias_bar[0] += update_bar[io];

      /* Derivative of weight application */
      for (int ii = 0; ii < dim_In; ii++)
      {
         if (compute_gradient) weights_bar[io*dim_In + ii] += state[ii] * update_bar[io];
         state_bar[ii] += weights[io*dim_In + ii] * update_bar[io]; 
      }
   }
}


OpenDenseLayer::OpenDenseLayer(int     dimI,
                               int     dimO,
                               int     Activ,
                               double  gammatik) : DenseLayer(0, dimI, dimO, 1.0, Activ, gammatik, 0.0) 
{
    type    = OPENDENSE;
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
                              double* state_bar,
                              int     compute_gradient)
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
   if (compute_gradient) 
   {
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
}                


OpenExpandZero::OpenExpandZero(int dimI,
                               int dimO) : Layer(0, OPENZERO, dimI, dimO, 1)
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
                              double* state_bar,
                              int     compute_gradient)
{
   for (int ii = 0; ii < dim_Out; ii++)
   {
      state_bar[ii] = 0.0;
   }
}                           


ClassificationLayer::ClassificationLayer(int    idx,
                                         int    dimI,
                                         int    dimO,
                                         double gammatik) : Layer(idx, CLASSIFICATION, dimI, dimO, dimO)
{
    gamma_tik = gammatik;
    /* Allocate the probability vector */
    probability = new double[dimO];
    tmpstate    = new double[dim_In];
}

ClassificationLayer::~ClassificationLayer()
{
    delete [] probability;
    delete [] tmpstate;
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
                                   double* state_bar,
                                   int     compute_gradient)
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
        if (compute_gradient) bias_bar[io] += update_bar[io];
  
        /* Derivative of weight application */
        for (int ii = 0; ii < dim_In; ii++)
        {
           if (compute_gradient) weights_bar[io*dim_In + ii] += state[ii] * update_bar[io];
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

double ClassificationLayer::crossEntropy(double *data_Out, 
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
      
      
void ClassificationLayer::crossEntropy_diff(double *data_Out, 
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

void ClassificationLayer::evalClassification(int      nexamples, 
                                             double** state,
                                             double** labels, 
                                             double*  loss_ptr, 
                                             double*  accuracy_ptr)
{
    double loss, accuracy;
    int    success;

    /* Sanity check */
    if (labels == NULL) printf("\n\n: ERROR: No labels for classification... \n\n");

    loss    = 0.0;
    success = 0;
    for (int iex = 0; iex < nexamples; iex++)
    {
        /* Copy values so that they are not overwrittn (they are needed for adjoint)*/
        for (int ic = 0; ic < dim_In; ic++)
        {
            tmpstate[ic] = state[iex][ic];
        }
        /* Apply classification on tmpstate */
        applyFWD(tmpstate);
        /* Evaluate Loss */
        loss     += crossEntropy(tmpstate, labels[iex]);
        success  += prediction(tmpstate, labels[iex]);
    }
    loss     = 1. / nexamples * loss;
    accuracy = 100.0 * (double) success / nexamples;
    // printf("%d: Eval loss %d,%1.14e using %1.14e\n", app->myid, ilayer, loss, u->state[1][1]);

    /* Return */
    *loss_ptr      = loss;
    *accuracy_ptr  = accuracy;

}       


void ClassificationLayer::evalClassification_diff(int      nexamples, 
                                                  double** primalstate,
                                                  double** adjointstate,
                                                  double** labels, 
                                                  int      compute_gradient)
{
    /* Recompute the Classification */
    for (int iex = 0; iex < nexamples; iex++)
    {
        /* Copy values into auxiliary vector */
        for (int ic = 0; ic < dim_In; ic++)
        {
            tmpstate[ic] = primalstate[iex][ic];
        }
        /* Apply classification on tmpstate */
        applyFWD(tmpstate);
    }
    
    /* Derivative of Loss and classification. This updates adjoint state and gradient, if desired. */
    double loss_bar = 1./nexamples; 
    for (int iex = 0; iex < nexamples; iex++)
    {
        crossEntropy_diff(tmpstate, adjointstate[iex], labels[iex], loss_bar);

        applyBWD(primalstate[iex], adjointstate[iex], compute_gradient);
    }
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