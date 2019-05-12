#include "defs.hpp"
#include "braid_wrapper.hpp"
#pragma once

/**
 * Abstract solver base class.
 */

class Solver
{
   protected:
      Config* config;
      MyReal objective;  /* Objective function value */

   public:

      /* Constructor */
      Solver(Config*  config);

      /* Destructor */
      ~Solver();

      /* Execute the solver in forward mode for objective function eval. */
      virtual MyReal runFWD(DataSet* data) = 0;

      /* Execute the solver in backward mode for gradient eval. */
      virtual MyReal runBWD(DataSet* data) = 0;

      /* Return network distribution to local processors */
      virtual void getGridDistribution(int* ilower_ptr, 
                                       int* iupper_ptr);

      /* Return objective function value */
      void getObjective(MyReal* obj_ptr);
};

/**
 * Braid Solver 
 */
class BraidSolver : public Solver
{
   protected: 
      myBraidApp        *primalapp;   /**< Primal app for cost function eval */
      myAdjointBraidApp *adjointapp;  /**< Adjoint app for gradient eval. */


   public:

      /* Constructor */
      BraidSolver(Config*   config, 
                  Network*  network, 
                  MPI_Comm comm);

      /* Destructor */
      ~BraidSolver();

      /* Run primal XBraid iterations */
      MyReal runFWD(DataSet* data);

      /* Run adjoint XBraid iterations */
      MyReal runBWD(DataSet* data);

      /* Get local grid distribution */
      void getGridDistribution(int* ilower_ptr, 
                               int* iupper_ptr);
};

// /**
//  * MLMC Solver
//  */
// class MLMCSolver : public Solver
// {
//       /* Constructor */
//       MLMCSolver();
//       /* Destructor */
//       ~MLMCSolver();

//       /* Run MLMC */
//       int run();
// };