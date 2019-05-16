#include "solver.hpp"

Solver::Solver()
{
   objective = -1.0;
}

Solver::~Solver(){}


void Solver::getObjective(MyReal* obj_ptr){ *obj_ptr = objective; }

BraidSolver::BraidSolver(Config*   config, 
                         Network*  network, 
                         MPI_Comm  comm) : Solver()
{
   /* Initialize XBraid */
   primalapp  = new myBraidApp(network, config, comm);
   adjointapp = new myAdjointBraidApp(network, config, primalapp->getCore(), comm);
}

BraidSolver::~BraidSolver()
{
    delete primalapp;
    delete adjointapp;
}

MyReal BraidSolver::runFWD(DataSet* data)
{
   /* Set the data */
   primalapp->data = data;
   /* Solve */
   double residual = primalapp->run();
   /* Update the objective function */
   primalapp->getObjective(&objective);

   return residual;
}

MyReal BraidSolver::runBWD(DataSet* data)
{
   /* Set data */
   adjointapp->data = data;
   /* Solve */
   double residual = adjointapp->run();

   return residual;
}

void BraidSolver::getGridDistribution(Config* config,
                                      int* ilower_ptr, 
                                      int* iupper_ptr)
{
   primalapp->getGridDistribution(ilower_ptr, iupper_ptr);
}

TimesteppingSolver::TimesteppingSolver(Config*  config,
                                       Network* net)
{
   network   = net;
   nchannels = network->getnChannels();

   /* Allocate the state */
   state = new MyReal[nchannels];
}

TimesteppingSolver::~TimesteppingSolver()
{
   delete [] state;
}

MyReal TimesteppingSolver::runFWD(DataSet* data)
{
   printf("\n Hi! I'm TimesteppingSolver. I'll runFWD() now!\n");
   
   /* MOVE ALL THIS INTO Network Class!! */

   Layer* layer;
   int class_id;
   double success_local;
   double loss     = 0.0;
   double success  = 0.0;
   double accuracy = 0.0;

   int startlayerID = network->getStartLayerID();
   int endlayerID   = network->getEndLayerID();
   if (startlayerID == 0) startlayerID -= 1; // this includes opening layer (id = -1) 


   /* Iterate over all examples */
   for (int iex = 0; iex < data->getnBatch(); iex++)
   {
      /* Iterate over all layers */
      for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
      {
         /* Get the layer */
         layer = network->getLayer(ilayer);

         /* Set example data at first layer */
         if (ilayer == -1) 
         {
            layer->setExample(data->getExample(iex));
         }

         /* Set label at last layer */
         if (ilayer == network->getnLayersGlobal()-2) 
         {
            layer->setLabel(data->getLabel(iex));
         }

         /* Apply the layer */
         layer->applyFWD(state);

         /* Evaluate Loss */
         if (ilayer == network->getnLayersGlobal()-2)
         {
            loss          += layer->crossEntropy(state);
            success_local  = layer->prediction(state, &class_id);
            success       += success_local;
         }
      }

      loss     = 1. / data->getnBatch() * loss;
      accuracy = 100.0 * ( (MyReal) success ) / data->getnBatch();
   }
     

   return -1.0;
}

MyReal TimesteppingSolver::runBWD(DataSet* data)
{
   printf("\n Hi! I'm TimesteppingSolver. I'll runBWD() now!\n");
   return -1.0;
}

void TimesteppingSolver::getGridDistribution(Config* config,
                                             int* ilower_ptr, 
                                             int* iupper_ptr)
{
   *ilower_ptr = 0;
   *iupper_ptr = config->nlayers - 2; 
}