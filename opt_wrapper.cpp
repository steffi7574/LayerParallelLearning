#include "opt_wrapper.hpp"

myFunction::myFunction()
{
   primaltrainapp  = NULL;
   adjointtrainapp = NULL;
   primalvalapp    = NULL;
}

myFunction::~myFunction(){}

void myFunction::initialize(myBraidApp*        PrimalTrain,
                            myAdjointBraidApp* AdjointTrain,
                            myBraidApp*        PrimalVal)
{
   primaltrainapp  = PrimalTrain;
   adjointtrainapp = AdjointTrain;
   primalvalapp    = PrimalVal;

   network = primaltrainapp->getNetwork();
}                         
      
void myFunction::setDesign(int size_design, MyReal* design_ptr)
{
   size_design = network->getnDesignLocal();
   design_ptr  = network->getDesign();
   
}

MyReal myFunction::evaluateObjective(MyReal* design)
{
   // network->getDesign() == design;


   return 0.0;
}

MyReal myFunction::evaluateGradient(MyReal* design, MyReal* gradient) 
{

   return 0.0;
}

void myFunction::callMeOncePerIter()
{}

// OptInterface::OptInterface(my_App*   myapptrain, 
//                            my_App*   myappvalid,
//                           braid_Core mycoretrain,
//                           braid_Core mycorevalid,
//                           FILE*      outputfile,
//                           MPI_Comm   mycomm)
// {
//   app_train  = myapptrain;
//   app_valid  = myappvalid;
//   core_train =  mycoretrain;
//   core_valid =  mycorevalid;

//   optimfile = outputfile;
//   comm_opt  = mycomm;
// }

// OptInterface::~OptInterface(){}


// bool OptInterface::get_prob_sizes(long long& n, long long& m)
// {
//   n = app_train->network->getnDesign();
//   m = 0;

//   return true;
// }

// bool OptInterface::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
// {
//   for (int idesign = 0; idesign < n; idesign++)
//   {
//     xlow[idesign] = -1e21;
//     xupp[idesign] = 1e21;
//     type[idesign] = hiopNonlinear;
//   }
//   return true;
// }

// bool OptInterface::get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
// {
//   return true;
// }

// bool OptInterface::eval_f(const long long& n, const double* x_in, bool new_x, double& obj_value)
// {
//   double objective;

//   /* Recompute objective, if new design is used. Otherwise, use stored objective function value */
//   if (new_x)
//   {
//     /* Set the network design variables */
//     for (int idesign = 0; idesign < n; idesign++)
//     {
//       app_train->network->getDesign()[idesign] = x_in[idesign];
//     }
    
//     /* Solve with braid */
//     app_train->loss     = 0.0;
//     app_train->accuracy = 0.0;
//     braid_SetObjectiveOnly(core_train, 1);
//     braid_SetPrintLevel(core_train, 0);
//     braid_Drive(core_train);
//     braid_GetObjective(core_train, &objective);

//     /* Buffer the objective function */
//     app_train->objective = objective;
//   }
//   else
//   {
//     objective = app_train->objective;
//   }

//   /* Return objective function to hiop */
//   obj_value = objective;

//   return true;
// }
   
// bool OptInterface::eval_grad_f(const long long& n, const double* x_in, bool new_x, double* gradf)
// {
//   int nreq = -1;
//   double gnorm, rnorm, rnorm_adj;
//   double objective;

//   /* Set the network design variables */
//   for (int idesign = 0; idesign < n; idesign++)
//   {
//     app_train->network->getDesign()[idesign] = x_in[idesign];
//     app_valid->network->getDesign()[idesign] = x_in[idesign];
//   }

//   /* Compute objective function and gradient with braid */
//   app_train->loss     = 0.0;
//   app_train->accuracy = 0.0;
//   braid_SetObjectiveOnly(core_train, 0);
//   braid_SetPrintLevel(core_train, 1);
//   braid_Drive(core_train);
//   braid_GetObjective(core_train, &objective);
//   app_train->objective = objective;

//   /* Return gradient to hiop */
//   MPI_Allreduce(app_train->network->getGradient(), gradf, app_train->network->getnDesign(), MPI_DOUBLE, MPI_SUM, app_train->comm_braid);

//   /* Get primal and adjoint residual norms */
//   braid_GetRNorms(core_train, &nreq, &rnorm);
//   braid_GetRNormAdjoint(core_train, &rnorm_adj);


//   /* Compute validation accuracy */
//   app_valid->loss     = 0.0;
//   app_valid->accuracy = 0.0;
//   braid_SetObjectiveOnly(core_valid, 1);
//   braid_Drive(core_valid);


//   /* Communicate loss and accuracy. This is actually not needed, except for printing output. Remove it. */
//   double train_loss, train_accur, val_accur;
//   MPI_Allreduce(&app_train->loss, &train_loss, 1, MPI_DOUBLE, MPI_SUM, app_train->comm_braid);
//   MPI_Allreduce(&app_train->accuracy, &train_accur, 1, MPI_DOUBLE, MPI_SUM, app_train->comm_braid);
//   MPI_Allreduce(&app_valid->accuracy, &val_accur, 1, MPI_DOUBLE, MPI_SUM, app_valid->comm_braid);
//   app_train->loss     = train_loss;
//   app_train->accuracy = train_accur;
//   app_valid->accuracy = val_accur;

//   /* Optimization output */
//   if (app_train->myid == 0)
//   {
//     printf("%1.8e  %1.8e  %1.14e  %1.14e        %2.2f%%      %2.2f%%\n", rnorm, rnorm_adj, objective, app_train->loss, app_train->accuracy, app_valid->accuracy);
//     fprintf(optimfile,"%1.8e  %1.8e  %1.14e  %1.14e        %2.2f%%      %2.2f%%\n", rnorm, rnorm_adj, objective, app_train->loss, app_train->accuracy, app_valid->accuracy);
//     fflush(optimfile);
//   }

//   return true;
// }

// bool OptInterface::eval_cons(const long long& n,
//                   const long long& m,
//                   const long long& num_cons, const long long* idx_cons,
//                   const double* x_in, bool new_x, double* cons)
// {
//   return true;
// }

// bool OptInterface::eval_Jac_cons(const long long& n, const long long& m,
//                       const long long& num_cons, const long long* idx_cons,
//                       const double* x_in, bool new_x, double** Jac)
// {
//   return true;
// }


// bool OptInterface::get_starting_point (const long long& n, double* x0)
// {
  
//   for (int idesign = 0; idesign < n; idesign++)
//   {
//     x0[idesign] = app_train->network->getDesign()[idesign];
//   }

//   return true;
// }


// bool OptInterface::get_MPI_comm(MPI_Comm& comm_out)
// {
//   comm_out = comm_opt;

//   return true;
// }
