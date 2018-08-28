#include "network.hpp"

Network::Network()
{
   nlayers     = 0;
   nchannels   = 0;
   dt          = 0.0;
   gamma_tik   = 0.0;
   gamma_ddt   = 0.0;
   loss        = 0.0;
   accuracy    = 0.0;

   openlayer   = NULL;
   layers      = NULL;
   endlayer    = NULL;

   state_curr  = NULL;
   state_old   = NULL;
   state_final = NULL;
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
                 double gammaTIK,
                 double gammaDDT)
{
   nlayers   = nLayers;
   nchannels = nChannels;
   dt        = deltaT;
   gamma_tik = gammaTIK;
   gamma_ddt = gammaDDT;
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

   /* Create and initialize the opening layer */
   if (Weight_open_init == 0.0)
   {
      openlayer = new OpenExpandZero(nFeatures, nChannels, deltaT);
   }
   else
   {
      openlayer = new DenseLayer(0, nFeatures, nChannels, deltaT, activ_ptr, dactiv_ptr);
   }
   openlayer->initialize(Weight_open_init);
   openlayer->setDt(1.0);

   /* Create and initialize the intermediate layers */
   layers = new Layer*[nlayers-2];
   for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
   {
      layers[ilayer-1] = new DenseLayer(ilayer, nChannels, nChannels, deltaT, activ_ptr, dactiv_ptr);
      layers[ilayer-1]->initialize(Weight_init);
   }

   /* Create and initialize the end layer */
   endlayer = new ClassificationLayer(nLayers, nChannels, nClasses, deltaT);
   endlayer->initialize(Classification_init);


   /* Allocate temporary vectors */
    state_curr  = new double[nChannels];
    state_old   = new double[nChannels];
    state_final = new double[nClasses];
}              

Network::~Network()
{
    /* Delete the layers */
    delete openlayer;
    for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
    {
       delete layers[ilayer-1];
    }
    delete [] layers;
    delete endlayer;

    delete [] state_curr;
    delete [] state_old;
    delete [] state_final;
}

int Network::getnChannels() { return nchannels; }
int Network::getnLayers()   { return nlayers; }

void Network::applyFWD(int      nexamples,
                       double **examples,
                       double **labels)
{
    int class_id = -1;
    int success  = 0;
    double objective  = 0.0;
    double regul_tikh = 0.0;
    double regul_ddt  = 0.0;


    /* Propagate the example */
    loss = 0.0;
    for (int iex = 0; iex < nexamples; iex++)
    {
        /* Apply opening layer */
        openlayer->applyFWD(examples[iex], state_curr);

        /* Evaluate regularization term */
        regul_tikh += openlayer->evalTikh();

        /* Propagate through all intermediate layers */ 
        for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
        {
            for (int ichannels = 0; ichannels < nchannels; ichannels++)
            {
                state_old[ichannels] = state_curr[ichannels];
            } 
            layers[ilayer-1]->applyFWD(state_old, state_curr);

            /* Evaluate regularization term */
            regul_tikh += layers[ilayer-1]->evalTikh();
            if (ilayer > 1) regul_ddt += evalRegulDDT(layers[ilayer-2], layers[ilayer-1]);
        }

        /* Apply classification layer */
        endlayer->applyFWD(state_curr, state_final);

        /* Evaluate regularization term */
        regul_tikh += endlayer->evalTikh();

        /* Evaluate loss */
        loss += endlayer->evalLoss(state_final, labels[iex]);

        /* Test for successful prediction */
        class_id = endlayer->prediction(state_final);
        if ( labels[iex][class_id] > 0.99 )  
        {
            success++;
        }
    }

    /* Compute objective function */
    loss       = 1. / nexamples * loss;
    objective  = loss + gamma_tik * regul_tikh + gamma_ddt * regul_ddt;

    /* Compute network accuracy */
    accuracy = 100.0 * (double) success / nexamples;


    /* Output */
    printf("Loss:      %1.14e\n",   loss);
    printf("Objective: %1.14e\n",   objective);
    printf("Accuracy:  %3.4f %%\n", accuracy);

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