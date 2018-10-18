#include "network.hpp"
#include<assert.h>

Network::Network()
{
   nlayers     = 0;
   nchannels   = 0;
   dt          = 0.0;
   loss        = 0.0;
   accuracy    = 0.0;
   gamma_ddt   = 0.0;
   gradient    = NULL;
   design      = NULL;
   layers      = NULL;
}

Network::Network(int    nLayers,
                 int    nChannels, 
                 int    nFeatures,
                 int    nClasses,
                 int    Activation,
                 double deltaT,
                 double Weight_init,
                 double Weight_open_init,
                 double Classification_init,
                 double Gamma_tik, 
                 double Gamma_ddt,
                 double Gamma_class,
                 int   networkType)
{
    double (*activ_ptr)(double x);
    double (*dactiv_ptr)(double x);

    /* Sanity check */
    if (nFeatures > nChannels ||
        nClasses  > nChannels)
    {
        printf("ERROR! Choose a wider netword!\n");
        printf(" -- nFeatures = %d\n", nFeatures);
        printf(" -- nChannels = %d\n", nChannels);
        printf(" -- nClasses = %d\n", nClasses);
        exit(1);
    }

    /* Initilizize */
    nlayers   = nLayers;
    nchannels = nChannels;
    dt        = deltaT;
    loss      = 0.0;
    accuracy  = 0.0;
    gamma_ddt = Gamma_ddt;


    /* Set the activation function */
    switch ( Activation )
    {
       case TANH:
          activ_ptr  = &Network::tanh_act;
          dactiv_ptr = &Network::dtanh_act;
          break;
       case RELU:
           activ_ptr  = &Network::ReLu_act;
           dactiv_ptr = &Network::dReLu_act;
          break;
       case SMRELU:
           activ_ptr  = &Network::SmoothReLu_act;
           dactiv_ptr = &Network::dSmoothReLu_act;
          break;
       default:
          printf("ERROR: You should specify an activation function!\n");
          printf("GO HOME AND GET SOME SLEEP!");
    }

    /* --- Create the layers --- */
    layers  = new Layer*[nlayers];
    ndesign = 0;
    /* opening layer */
    switch ( networkType )
    {
       case DENSE:
          if (Weight_open_init == 0.0)
          {
             layers[0]  = new OpenExpandZero(nFeatures, nChannels);
          }
          else
          {
             layers[0] = new OpenDenseLayer(nFeatures, nChannels, activ_ptr, dactiv_ptr, Gamma_tik);
          }
          break;
       case CONVOLUTIONAL:
          /**< (Weight_open_init == 0.0) not needed for convolutional layers*/
          {
             layers[0] = new OpenConvLayer(nFeatures, nChannels);
          }
          break;
    }
    ndesign += layers[0]->getnDesign();

    /* intermediate layers */
    switch ( networkType )
    {
       case DENSE:
          for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
          {
             layers[ilayer] = new DenseLayer(ilayer, nChannels, nChannels, deltaT, activ_ptr, dactiv_ptr, Gamma_tik);
             ndesign += layers[ilayer]->getnDesign();
          }
          break;
       case CONVOLUTIONAL:
          for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
          {
             // TODO: Fix
             int convolution_size = 5;
             layers[ilayer] = new ConvLayer(ilayer, nChannels, nChannels,
                                            convolution_size,nChannels/nFeatures, 
                                            deltaT, activ_ptr, dactiv_ptr, Gamma_tik);
             ndesign += layers[ilayer]->getnDesign();
          }
          break;
    }

    /* end layer */
    layers[nlayers-1] = new ClassificationLayer(nLayers-1, nChannels, nClasses, Gamma_class);
    ndesign += layers[nlayers-1]->getnDesign();
 
    /* Allocate memory for network design and gradient variables */
    design   = new double[ndesign];
    gradient = new double[ndesign];

    /* Initialize  the layer weights and bias */
    int istart = 0;
    layers[0]->initialize(&(design[istart]), &(gradient[istart]), Weight_open_init);
    istart += layers[0]->getnDesign();
    for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
    {
        layers[ilayer]->initialize(&(design[istart]), &(gradient[istart]), Weight_init);
        istart += layers[ilayer]->getnDesign();
    }
    layers[nlayers-1]->initialize(&(design[istart]), &(gradient[istart]), Classification_init);

}             

  

Network::~Network()
{
    /* Delete the layers */
    for (int ilayer = 0; ilayer < nlayers; ilayer++)
    {
       delete layers[ilayer];
    }
    delete [] layers;

    delete [] design;
    delete [] gradient;
}

int Network::getnChannels() { return nchannels; }

int Network::getnLayers() { return nlayers; }

double Network::getLoss() { return loss; }

double Network::getAccuracy() { return accuracy; }

int Network::getnDesign() { return ndesign; }

double* Network::getDesign() { return design; }
       
double* Network::getGradient() { return gradient; }

void Network::applyFWD(int      nexamples,
                       double **examples,
                       double **labels)
{
    int success;
    double* state = new double[nchannels];

    /* Propagate the examples */
    loss    = 0.0;
    success = 0;
    for (int iex = 0; iex < nexamples; iex++)
    { 
        /* Load input data */
        layers[0]->setExample(examples[iex]);
       
        /* Propagate through all layers */ 
        for (int ilayer = 0; ilayer < nlayers; ilayer++)
        {
            /* Apply the next layer */
            layers[ilayer]->applyFWD(state);
        }

        /* Evaluate loss */
        loss += layers[nlayers-1]->evalLoss(state, labels[iex]);

        /* Test for successful prediction */
        success += layers[nlayers-1]->prediction(state, labels[iex]);
    }
        
    /* Set loss and accuracy */
    loss     = 1. / nexamples * loss;
    accuracy = 100.0 * (double) success / nexamples;

    delete [] state;
}


double Network::evalRegularization()
{
    double regul_tikh  = 0.0;
    double regul_ddt   = 0.0;

    /* Evaluate regularization terms for each layer */
    for (int ilayer = 0; ilayer < nlayers; ilayer++)
    {
        regul_tikh += layers[ilayer]->evalTikh();
        if (ilayer > 1 && ilayer < nlayers - 1) regul_ddt += evalRegulDDT(layers[ilayer-1], layers[ilayer]);
    }

    return regul_tikh + regul_ddt;
}


double Network::evalRegulDDT(Layer* layer_old, 
                             Layer* layer_curr)
{
    double diff;
    double ddt = 0.0;

    /* Sanity check */
    if (layer_old->getDimIn()    != nchannels ||
        layer_old->getDimOut()   != nchannels ||
        layer_old->getDimBias()  != 1         ||
        layer_curr->getDimIn()   != nchannels ||
        layer_curr->getDimOut()  != nchannels ||
        layer_curr->getDimBias() != 1           )
        {
            printf("ERROR when evaluating ddt-regularization of intermediate Layers.\n"); 
            printf("Dimensions don't match. Check and change this routine.\n");
            exit(1);
        }
   
    assert(layer_curr->getnDesign()==layer_old->getnDesign());
    int nDesign = layer_curr->getnDesign();

    for (int iw = 0; iw < nDesign; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / dt;
        ddt += pow(diff,2);
    }
    diff = (layer_curr->getBias()[0] - layer_old->getBias()[0]) / dt;
    ddt += pow(diff,2);
    
    return gamma_ddt / 2.0 * ddt;
}                

void Network::evalRegulDDT_diff(Layer* layer_old, 
                                Layer* layer_curr,
                                double regul_bar)
{
    double diff;
    regul_bar = gamma_ddt * regul_bar;

    /* Derivative of the bias-term */
    diff = (layer_curr->getBias()[0] - layer_old->getBias()[0]) / pow(dt,2);
    layer_curr->getBiasBar()[0] += diff * regul_bar;
    layer_old->getBiasBar()[0]  -= diff * regul_bar;

    assert(layer_curr->getnDesign()==layer_old->getnDesign());
    int nDesign = layer_curr->getnDesign();

    /* Derivative of the weights term */
    for (int iw = 0; iw < nDesign; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / pow(dt,2);
        layer_curr->getWeightsBar()[iw] += diff * regul_bar;
        layer_old->getWeightsBar()[iw]  -= diff * regul_bar;
    }
} 

double Network::ReLu_act(double x)
{
    return std::max(0.0, x);
}


double Network::dReLu_act(double x)
{
    double diff;
    if (x >= 0.0) diff = 1.0;
    else         diff = 0.0;

    return diff;
}


double Network::SmoothReLu_act(double x)
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
        return Network::ReLu_act(x);
    }
}

double Network::dSmoothReLu_act(double x)
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
        return Network::dReLu_act(x);
    }

}


double Network::tanh_act(double x)
{
    return tanh(x);
}

double Network::dtanh_act(double x)
{
    double diff = 1.0 - pow(tanh(x),2);

    return diff;
}
