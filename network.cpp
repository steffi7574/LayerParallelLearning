#include "network.hpp"

Network::Network()
{
   nlayers_global = 0;
   nlayers_local  = 0;
   nchannels      = 0;
   dt             = 0.0;
   loss           = 0.0;
   accuracy       = 0.0;
   gamma_tik      = 0.0;
   gamma_ddt      = 0.0;
   gamma_class    = 0.0;
   gradient       = NULL;
   design         = NULL;
   layers         = NULL;
}

Network::Network(int    nLayersGlobal,
                 int    nChannels, 
                 double deltaT,
                 double Gamma_tik, 
                 double Gamma_ddt,
                 double Gamma_class)
{

    /* Initilizize */
    nlayers_global   = nLayersGlobal;
    nchannels        = nChannels;
    dt               = deltaT;
    loss             = 0.0;
    accuracy         = 0.0;
    gamma_tik        = Gamma_tik;
    gamma_ddt        = Gamma_ddt;
    gamma_class      = Gamma_class;

}             

  



Network::~Network()
{
    /* Delete the layers */
    for (int ilayer = 0; ilayer < nlayers_local; ilayer++)
    {
        delete layers[ilayer];
    }
    delete [] layers;

    delete [] design;
    delete [] gradient;
}

int Network::getnChannels() { return nchannels; }

int Network::getnLayers() { return nlayers_global; }

double Network::getDT() { return dt; }

int Network::getLocalID(int ilayer) 
{
    int idx = ilayer - startlayerID;
    if (idx < 0 || idx > nlayers_global-1) 
    {
           printf("\n\nERROR! Something went wrong with local storage of layers! \n");
           printf("ilayer %d, startlayerID %d\n\n", ilayer, startlayerID);
    }

    return idx;
}

double Network::getLoss() { return loss; }

double Network::getAccuracy() { return accuracy; }

int Network::getnDesign() { return ndesign; }

double* Network::getDesign() { return design; }
       
double* Network::getGradient() { return gradient; }





void Network::createLayers(int    StartLayerID, 
                           int    EndLayerID, 
                           int    nFeatures,
                           int    nClasses,
                           int    Activation,
                           double Weight_init,
                           double Weight_open_init,
                           double Classification_init)
{

    startlayerID = StartLayerID;
    endlayerID   = EndLayerID;
    nlayers_local = endlayerID - startlayerID + 1;


    /* Sanity check */
    if (nFeatures > nchannels ||
        nClasses  > nchannels)
    {
        printf("ERROR! Choose a wider netword!\n");
        exit(1);
    }

    printf("creating layers startid %d endid %d, nlayer_local %d\n", startlayerID, endlayerID, nlayers_local);

   /* --- Create the layers --- */
    layers  = new Layer*[nlayers_local];
    ndesign = 0;
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        /* Create a layer at time step ilayer. Local storage at ilayer - startlayerID */
        int storeID = getLocalID(ilayer);
        if (ilayer == 0)  // Opening layer
        {
            if (Weight_open_init == 0.0)
            {
               layers[storeID]  = new OpenExpandZero(nFeatures, nchannels);
               printf("Creating OpenExpandZero-Layer at %d local %d\n", ilayer, storeID);
            }
            else
            {
               layers[storeID] = new OpenDenseLayer(nFeatures, nchannels, Activation, gamma_tik);
               printf("Creating OpenDense-Layer at %d local %d\n", ilayer, storeID);
            }
            ndesign += layers[storeID]->getnDesign();
        }
        else if (ilayer < nlayers_global-1) // Intermediate layer
        {
            layers[storeID] = new DenseLayer(ilayer, nchannels, nchannels, dt, Activation, gamma_tik);
            ndesign += layers[storeID]->getnDesign();
            printf("Creating Dense-Layer at %d local %d\n", ilayer, storeID);
        }
        else // Classification layer 
        {
            layers[storeID] = new ClassificationLayer(nlayers_global-1, nchannels, nClasses, gamma_class);
            ndesign += layers[storeID]->getnDesign();
            printf("Creating Classification-Layer at %d local %d\n", ilayer, storeID);
        }
    }

    /* Allocate memory for network design and gradient variables */
    design   = new double[ndesign];
    gradient = new double[ndesign];

    /* Initialize  the layer weights and bias */
    int istart = 0;
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        int storeID = ilayer - startlayerID;
        if (ilayer == 0)  // Opening layer
        {
            layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), Weight_open_init);
            istart += layers[storeID]->getnDesign();
        }
        else if (ilayer < nlayers_global-1) // Intermediate layer
        {
            layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), Weight_init);
            istart += layers[storeID]->getnDesign();
        }
        else // Classification layer 
        {
            layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), Classification_init);
        }
    }
}    





// void Network::applyFWD(int      nexamples,
//                        double **examples,
//                        double **labels)
// {
//     int success;
//     double* state = new double[nchannels];

//     /* Propagate the examples */
//     loss    = 0.0;
//     success = 0;
//     for (int iex = 0; iex < nexamples; iex++)
//     { 
//         /* Load input data */
//         layers[0]->setExample(examples[iex]);
       
//         /* Propagate through all layers */ 
//         for (int ilayer = 0; ilayer < nlayers; ilayer++)
//         {
//             /* Apply the next layer */
//             layers[ilayer]->applyFWD(state);
//         }

//         /* Evaluate loss */
//         loss += layers[nlayers-1]->evalLoss(state, labels[iex]);

//         /* Test for successful prediction */
//         success += layers[nlayers-1]->prediction(state, labels[iex]);
//     }
        
//     /* Set loss and accuracy */
//     loss     = 1. / nexamples * loss;
//     accuracy = 100.0 * (double) success / nexamples;

//     delete [] state;
// }


// double Network::evalRegularization()
// {
//     double regul_tikh  = 0.0;
//     double regul_ddt   = 0.0;

//     /* Evaluate regularization terms for each layer */
//     for (int ilayer = 0; ilayer < nlayers; ilayer++)
//     {
//         regul_tikh += layers[ilayer]->evalTikh();
//         if (ilayer > 1 && ilayer < nlayers - 1) regul_ddt += evalRegulDDT(layers[ilayer-1], layers[ilayer]);
//     }

//     return regul_tikh + regul_ddt;
// }


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

    /* Derivative of the weights term */
    for (int iw = 0; iw < nchannels * nchannels; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / pow(dt,2);
        layer_curr->getWeightsBar()[iw] += diff * regul_bar;
        layer_old->getWeightsBar()[iw]  -= diff * regul_bar;
    }
} 

