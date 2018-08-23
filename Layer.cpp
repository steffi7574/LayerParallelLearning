#include "Layer.hpp"

Layer::Layer()
{
   nchannels   = 0;
   dt          = 0.0;
   weights     = NULL;
   weights_bar = NULL;
   bias        = NULL;
   bias_bar    = NULL;
   activation  = NULL;
   update      = NULL;
   update_bar  = NULL;
}


Layer::Layer(int    nChannels,
             double (*Activ)(double x),
             double (*dActiv)(double x))
{
   nchannels   = nChannels;
   activation  = Activ;
   dactivation = dActiv;

   update     = new double[nchannels];
   update_bar = new double[nchannels];
}   


Layer::~Layer()
{
   delete [] update;
   delete [] update_bar;
}

void Layer::setBias(double* bias_ptr)
{
   bias = bias_ptr;
}

void Layer::setBias_bar(double* bias_bar_ptr)
{
   bias_bar = bias_bar_ptr;
}


void Layer::setWeights(double* weights_ptr)
{
   weights = weights_ptr;
}

void Layer::setWeights_bar(double* weights_bar_ptr)
{
   weights_bar = weights_bar_ptr;
}

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
      update[ichannel] += bias[0];

      /* apply activation */
      update[ichannel] = activation(update[ichannel]);
   }

   /* Update */
   for (int ichannel = 0; ichannel < nchannels; ichannel++)
   {
      data[ichannel] += dt * update[ichannel];
   }

}


void Layer::applyBWD(double* data, 
                     double* data_bar)
{
   /* Apply derivative of the update step */
   for (int ichannel = 0; ichannel < nchannels; ichannel++)
   {
      update_bar[ichannel] = dt * data_bar[ichannel];
   }

   /* Backward propagation for each channel */
   for (int ichannel = 0; ichannel < nchannels; ichannel++)
   {
      /* Recompute udate[ichannel] */
      update[ichannel] = vecdot(nchannels, &(weights[ichannel*nchannels]), data);

      /* Derivative of activation function */
      update_bar[ichannel] = dactivation(update[ichannel]) * update_bar[ichannel];

      /* Derivative of bias addition */
      bias_bar[0] += update_bar[ichannel];

      /* Derivative of weight application */
      for (int jchannel = 0; jchannel < nchannels; jchannel++)
      {
         data_bar[jchannel] += weights[ichannel*nchannels + jchannel] * update_bar[ichannel];
         weights_bar[ichannel*nchannels + jchannel] += data[jchannel] * update_bar[ichannel];
      }
   }
}                  