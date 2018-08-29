#include "network.hpp"

Network::Network()
{
   nlayers     = 0;
   nchannels   = 0;
   dt          = 0.0;
   loss        = 0.0;
   accuracy    = 0.0;
   state_upd   = NULL;
   state       = NULL;
   state_bar   = NULL;
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
                 double Classification_init)
{
   nlayers   = nLayers;
   nchannels = nChannels;
   dt        = deltaT;
   loss      = 0.0;
   accuracy  = 0.0;
   
   double (*activ_ptr)(double x);
   double (*dactiv_ptr)(double x);

   /* Set the activation function */
   switch ( Activation )
   {
      case RELU:
          activ_ptr  = &Network::ReLu_act;
          dactiv_ptr = &Network::dReLu_act;
         break;
      case TANH:
         activ_ptr  = &Network::tanh_act;
         dactiv_ptr = &Network::dtanh_act;
         break;
      default:
         printf("ERROR: You should specify an activation function!\n");
         printf("GO HOME AND GET SOME SLEEP!");
   }

   /* --- Create and initialize the layers --- */
   layers = new Layer*[nlayers];

   /* opening layer */
   if (Weight_open_init == 0.0)
   {
      layers[0]  = new OpenExpandZero(nFeatures, nChannels, deltaT);
   }
   else
   {
      layers[0] = new DenseLayer(0, nFeatures, nChannels, deltaT, activ_ptr, dactiv_ptr);
   }
   layers[0]->initialize(Weight_open_init);
   layers[0]->setDt(1.0);

   /* intermediate layers */
   for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
   {
      layers[ilayer] = new DenseLayer(ilayer, nChannels, nChannels, deltaT, activ_ptr, dactiv_ptr);
      layers[ilayer]->initialize(Weight_init);
   }

   /* end layer */
   layers[nlayers-1] = new ClassificationLayer(nLayers-1, nChannels, nClasses, deltaT);
   layers[nlayers-1]->initialize(Classification_init);


   /* Allocate vectors for current primal and adjoint state and update of the network */
    state     = new double[nChannels];
    state_bar = new double[nChannels];
    state_upd = new double[nChannels];

    /* Sanity check */
    if (nFeatures > nChannels ||
        nClasses  > nChannels)
    {
        printf("ERROR! Choose a wider netword!\n");
        exit(1);
    }
}              

Network::~Network()
{
    /* Delete the layers */
    for (int ilayer = 0; ilayer < nlayers; ilayer++)
    {
       delete layers[ilayer];
    }
    delete [] layers;

    delete [] state;
    delete [] state_bar;
    delete [] state_upd;
}

int Network::getnChannels() { return nchannels; }

int Network::getnLayers() { return nlayers; }

double Network::getLoss() { return loss; }

double Network::getAccuracy() { return accuracy; }

void Network::setState(int     dimN, 
                       double* data)
{
    if (dimN > nchannels) 
    {
        printf("ERROR! This should never be the case...\n");
        exit(1);
    }

    for (int is = 0; is < dimN; is++)
    {
        state[is] = data[is];
    }
    for (int is = dimN; is < nchannels; is++)
    {
        state[is] = 0.0;
    }
}

double* Network::getState() { return state; }

void Network::setState_bar(double value)
{
    for (int is = 0; is < nchannels; is++)
    {
        state_bar[is] = value;
    }
}

double* Network::getState_bar() { return state_bar; }


void Network::applyFWD(int      nexamples,
                       double **examples,
                       double **labels)
{
    int class_id, success;

    /* Propagate the examples */
    loss    = 0.0;
    success = 0;
    for (int iex = 0; iex < nexamples; iex++)
    { 
        /* Load input data */
        setState(layers[0]->getDimIn(), examples[iex]);
       
        /* Propagate through all layers */ 
        for (int ilayer = 0; ilayer < nlayers; ilayer++)
        {
            /* Apply the next layer */
            layers[ilayer]->applyFWD(state, state_upd);

            /* Shift state_upd into state */
            setState(layers[ilayer]->getDimOut(), state_upd);
        }

        /* Evaluate loss */
        loss += layers[nlayers-1]->evalLoss(state, labels[iex]);

        /* Test for successful prediction */
        class_id = layers[nlayers-1]->prediction(state_upd);
        if ( labels[iex][class_id] > 0.99 )  
        {
            success++;
        }
    }
        
    /* Set loss and accuracy */
    loss     = 1. / nexamples * loss;
    accuracy = 100.0 * (double) success / nexamples;
}


double Network::evalRegularization(double gamma_tik,
                                   double gamma_ddt)
{
    double regul_tikh = 0.0;
    double regul_ddt  = 0.0;

    /* Evaluate regularization terms for each layer */
    for (int ilayer = 0; ilayer < nlayers; ilayer++)
    {
        regul_tikh += layers[ilayer]->evalTikh();
        if (ilayer > 1 && ilayer < nlayers - 1) regul_ddt += evalRegulDDT(layers[ilayer-1], layers[ilayer]);
    }

    return gamma_tik * regul_tikh + gamma_ddt * regul_ddt;
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

    for (int iw = 0; iw < nchannels * nchannels; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / dt;
        ddt += pow(diff,2);
    }
    diff = (layer_curr->getBias()[0] - layer_old->getBias()[0]) / dt;
    ddt += pow(diff,2);
    
    return ddt/2.0;
}                

double Network::ReLu_act(double x)
{
    return std::max(0.0, x);
}


double Network::dReLu_act(double x)
{
    double diff;
    if (x > 0.0) diff = 1.0;
    else         diff = 0.0;

    return diff;
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