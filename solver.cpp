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

MLMCSolver::MLMCSolver(Config*  conf,
                       Network* net)
{
   network = net;
   config  = conf; 
}

MLMCSolver::~MLMCSolver(){}

MyReal MLMCSolver::runFWD(DataSet* data)
{
   printf("\n Hi! I'm MLMCSolver. I'll runFWD() now!\n");
   return -1.0;
}

MyReal MLMCSolver::runBWD(DataSet* data)
{
   printf("\n Hi! I'm MLMCSolver. I'll runBWD() now!\n");
   return -1.0;
}

void MLMCSolver::getGridDistribution(Config* config,
                                     int* ilower_ptr, 
                                     int* iupper_ptr)
{
   *ilower_ptr = 0;
   *iupper_ptr = config->nlayers - 2; 
}