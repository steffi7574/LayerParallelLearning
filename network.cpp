#include "network.hpp"

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
                 double Gamma_class)
{
    /* Sanity check */
    if (nFeatures > nChannels ||
        nClasses  > nChannels)
    {
        printf("ERROR! Choose a wider netword!\n");
        exit(1);
    }

    /* Initilizize */
    nlayers   = nLayers;
    nchannels = nChannels;
    dt        = deltaT;
    loss      = 0.0;
    accuracy  = 0.0;
    gamma_ddt = Gamma_ddt;



    /* --- Create the layers --- */
    layers  = new Layer*[nlayers];
    ndesign = 0;
    /* opening layer */
    if (Weight_open_init == 0.0)
    {
       layers[0]  = new OpenExpandZero(nFeatures, nChannels);
    }
    else
    {
       layers[0] = new OpenDenseLayer(nFeatures, nChannels, Activation, Gamma_tik);
    }
    ndesign += layers[0]->getnDesign();
    /* intermediate layers */
    for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
    {
       layers[ilayer] = new DenseLayer(ilayer, nChannels, nChannels, deltaT, Activation, Gamma_tik);
       ndesign += layers[ilayer]->getnDesign();
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
    if (layer_old->getnDesign()  != layer_curr->getnDesign() ||
        layer_old->getDimIn()   != layer_curr->getDimIn()  ||
        layer_old->getDimOut()   != layer_curr->getDimOut()  ||
        layer_old->getDimBias()  != layer_curr->getDimBias() )
        {
            printf("ERROR when evaluating ddt-regularization of intermediate Layers.\n"); 
            printf("Dimensions don't match. Check and change this routine.\n");
            exit(1);
        }

    int nweights = layer_curr->getnDesign() - layer_curr->getDimBias();
    for (int iw = 0; iw < nweights; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / dt;
        ddt += pow(diff,2);
    }
    int nbias = layer_curr->getnDesign() - layer_curr->getDimIn() * layer_curr->getDimOut();
    for (int ib = 0; ib < nbias; ib++)
    {
        diff = (layer_curr->getBias()[ib] - layer_old->getBias()[ib]) / dt;
    }
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
    int nbias = layer_curr->getnDesign() - layer_curr->getDimIn() * layer_curr->getDimOut();
    for (int ib = 0; ib < nbias; ib++)
    {
        diff = (layer_curr->getBias()[ib] - layer_old->getBias()[ib]) / pow(dt,2);
        layer_curr->getBiasBar()[ib] += diff * regul_bar;
        layer_old->getBiasBar()[ib]  -= diff * regul_bar;
    }

    /* Derivative of the weights term */
    int nweights = layer_curr->getnDesign() - layer_curr->getDimBias();
    for (int iw = 0; iw < nweights; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / pow(dt,2);
        layer_curr->getWeightsBar()[iw] += diff * regul_bar;
        layer_old->getWeightsBar()[iw]  -= diff * regul_bar;
    }
} 

