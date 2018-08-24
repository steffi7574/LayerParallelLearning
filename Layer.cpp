#include "Layer.hpp"

Layer::Layer()
{
   nchannels    = 0;
   dt           = 0.0;
   weights      = NULL;
   weights_bar  = NULL;
   bias         = NULL;
   bias_bar     = NULL;
   activation   = NULL;
   dactivation  = NULL;
   update       = NULL;
   update_bar   = NULL;
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

void Layer::setInputData(double* inputdata_ptr){}


DenseLayer::DenseLayer(int    nChannels, 
                       double (*Activ)(double x),
                       double (*dActiv)(double x)) : Layer(nChannels, Activ, dActiv)
{
   /* Everything is done in Layer constructor */
}   

DenseLayer::~DenseLayer() {}


void DenseLayer::applyFWD(double* data)
{
   /* Compute update for each channel */
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


void DenseLayer::applyBWD(double* data, 
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
      update[ichannel]  = vecdot(nchannels, &(weights[ichannel*nchannels]), data);
      update[ichannel] += bias[0];

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


OpenLayer::OpenLayer(int nChannels,
                     int nFeatures, 
                     double (*Activ)(double x),
                     double (*dActiv)(double x)) : Layer(nChannels, Activ, dActiv)
{
   nfeatures = nFeatures;
}



OpenLayer::~OpenLayer(){}


void OpenLayer::setInputData(double* inputdata_ptr)
{
   inputData = inputdata_ptr;
}

void OpenLayer::applyFWD(double* data)
{
   /* If activation is not set, just expand the data to the network width using zeros */
   if (activation == NULL)
   {
      for (int ichannel = 0; ichannel < nchannels; ichannel++)
      {
         if (ichannel < nfeatures) data[ichannel] = inputData[ichannel];
         else                      data[ichannel] = 0.0;
      }
   }
   else
   {
      /* If activation is set, apply sigma(Ky+b) for each channel */
      for (int ichannel = 0; ichannel < nchannels; ichannel++)
      {
         /* Apply weights */
         data[ichannel] = vecdot(nfeatures, &(weights[ichannel*nfeatures]), inputData);

         /* Add bias */
         data[ichannel] += bias[0];

         /* apply activation */
         data[ichannel] = activation(data[ichannel]);
      }
   }
}


void OpenLayer::applyBWD(double* data, 
                         double* data_bar)
{
   if (activation == NULL)
   {
      /* Do nothing */
   }
   else
   {
      /* Backward propagation for each channel */
      for (int ichannel = 0; ichannel < nchannels; ichannel++)
      {
         /* Recompute update[ichannel] */
         update[ichannel]  = vecdot(nfeatures, &(weights[ichannel*nfeatures]), inputData);
         update[ichannel] += bias[0];

         /* Derivative of activation function */
         data_bar[ichannel] = dactivation(update[ichannel]) * data_bar[ichannel];

         /* Derivative of bias addition */
         bias_bar[0] += data_bar[ichannel];

         /* Derivative of weight application */
         for (int jfeatures = 0; jfeatures < nfeatures; jfeatures++)
         {
            weights_bar[ichannel*nfeatures + jfeatures] += inputData[jfeatures] * data_bar[ichannel];
         }
      } 
   }
}                      
