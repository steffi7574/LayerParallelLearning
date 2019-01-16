#include "optimizer.hpp"
#include "defs.hpp"
#include "braid_wrapper.hpp"
#pragma once


class myFunction: public UserFunction 
{
   protected: 
      myBraidApp*         primaltrainapp;
      myAdjointBraidApp*  adjointtrainapp;
      myBraidApp*         primalvalapp;

      Network*  network;

   public:
      myFunction();    /* Default constructor */
      ~myFunction();   

      /* Set the pointers */
      void initialize(myBraidApp*        PrimalTrain,
                      myAdjointBraidApp* AdjointTrain,
                      myBraidApp*        PrimalVal);


      /* Pass a pointer to the design to the optimizer */
      void setDesign(int size_design, MyReal* design_ptr);

      MyReal evaluateObjective(MyReal* design);
      MyReal evaluateGradient(MyReal* design, MyReal* gradient);
      void   callMeOncePerIter();
};




// class OptInterface : public hiop::hiopInterfaceDenseConstraints
// {
//    my_App*    app_train;
//    my_App*    app_valid;
//    braid_Core core_train;
//    braid_Core core_valid;
//    FILE*      optimfile;
//    MPI_Comm   comm_opt;

//    public:
//       OptInterface(my_App* myapptrain, my_App* myappvalid, braid_Core mycoretrain, braid_Core mycorevalid, FILE* outputfile, MPI_Comm mycomm);
//       virtual ~OptInterface();
   
   
//       bool get_prob_sizes(long long& n, long long& m);
   
//       bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type);
   
//       bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type);
   
//       bool eval_f(const long long& n, const double* x_in, bool new_x, double& obj_value);
      
//       bool eval_grad_f(const long long& n, const double* x_in, bool new_x, double* gradf);
   
//       bool eval_cons(const long long& n,
//                      const long long& m,
//                      const long long& num_cons, const long long* idx_cons,
//                      const double* x_in, bool new_x, double* cons);
   
//       bool eval_Jac_cons(const long long& n, const long long& m,
//                          const long long& num_cons, const long long* idx_cons,
//                          const double* x_in, bool new_x, double** Jac);

//       bool get_starting_point (const long long& n, double* x0);

//       bool get_MPI_comm(MPI_Comm& comm_out);

// }; 
