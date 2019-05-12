#include "solver.hpp"

Solver::Solver(Config*  conf)
{
   config    = conf;
   objective = -1.0;
}

Solver::~Solver(){}

void Solver::getGridDistribution(int* ilower_ptr, 
                                 int* iupper_ptr)
{
   *ilower_ptr = 0;
   *iupper_ptr = config->nlayers;

   printf("\n\n ERROR: CHECK GETGRIDDISTRIBUTION()!\n\n");
}  

void Solver::getObjective(MyReal* obj_ptr){ *obj_ptr = objective; }

BraidSolver::BraidSolver(Config*   config, 
                         Network*  network, 
                         MPI_Comm  comm) : Solver(config)
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

void BraidSolver::getGridDistribution(int* ilower_ptr, 
                                      int* iupper_ptr)
{
   primalapp->getGridDistribution(ilower_ptr, iupper_ptr);
}
// MLMCSolver::MLMCSolver()
// {

// }

// MLMCSolver::~MLMCSolver(){}

// int MLMCSolver::run()
// {
//    printf("\n Hi! I'm MLMCSolver. I'll run() now!\n");
//    return 0;
// }