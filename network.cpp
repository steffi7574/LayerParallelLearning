#include "network.hpp"

Network::Network()
{
   nlayers  = 0;
   layers   = NULL;
   endlayer = NULL;
}

Network::Network(int    nLayers,
                 int    nExamples,
                 int    nChannels, 
                 int    nFeatures,
                 int    nClasses,
                 int    Activation,
                 double Weight_init,
                 double Weight_open_init,
                 double Classification_init)
{
   nlayers   = nLayers;
   nexamples = nExamples;
   
   double (*activation)(double x);
   double (*dactivation)(double x);

   switch ( Activation )
   {
      case RELU:
          activation  = &Network::ReLu_act;
          dactivation = &Network::dReLu_act;
         break;
      case TANH:
         activation  = &Network::tanh_act;
         dactivation = &Network::dtanh_act;
         break;
      default:
         printf("ERROR: You should specify an activation function!\n");
         printf("GO HOME AND GET SOME SLEEP!");
   }

   /* Create the opening layer */
   if (Weight_open_init == 0.0)
   {
      openlayer = new OpenExpandZero(nFeatures, nChannels);
   }
   else
   {
      openlayer = new DenseLayer(0, nFeatures, nChannels, activation, dactivation);
   }

   /* Create the intermediate layers */
   layers = new Layer*[nlayers-2];
   for (int ilayer = 1; ilayer < nlayers-1; ilayer++)
   {
      layers[ilayer-1] = new DenseLayer(ilayer, nChannels, nChannels, activation, dactivation);
   }

   /* Create the end layer */
   endlayer = new ClassificationLayer(nLayers, nChannels, nClasses);
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