#include "Layer.hpp"

Layer::Layer()
{
   nchannels  = 0;
   dt         = 0.0;
   weights    = NULL;
   bias       = 0;
   activation = NULL;
   update     = NULL;
}


Layer::Layer(int    nChannels,
             double (*Activ)(double x))
{
   nchannels  = nChannels;
   activation = Activ;

   update = new double[nchannels];
}   


Layer::~Layer()
{
   delete [] update;
}

/* Set the bias */
void Layer::setBias(double Bias)
{
   bias = Bias;
}

/* Set pointer to the weight vector (matrix) */
void Layer::setWeights(double* Weights)
{
   weights = Weights;
}

/* Set time step size */
void Layer::setDt(double DT)
{
   dt = DT;
}


void Layer::applyFWD(double* data)
{
   /* Apply layer update for each channel */
   for (int ichannel = 0; ichannel < nchannels; ichannel++)
   {
      /* Apply weights */
      update[ichannel] = vecdot(nchannels, &(weights[ichannel*nchannels]), data);

      /* Add bias */
      update[ichannel] += bias;

      /* apply activation */
      update[ichannel] = activation(update[ichannel]);
   }

   /* Update */
   for (int ichannel = 0; ichannel < nchannels; ichannel++)
   {
      data[ichannel] += dt * update[ichannel];
   }


}

