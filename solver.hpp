#include "defs.hpp"
#include "braid_wrapper.hpp"
#pragma once

/**
 * Abstract solver base class.
 */

class Solver
{
   protected:
      MyReal objective;  /* Objective function value */

   public:

      /* Constructor */
      Solver();

      /* Destructor */
      ~Solver();

      /* Execute the solver in forward mode for objective function eval. */
      virtual MyReal runFWD(DataSet* data) = 0;

      /* Execute the solver in backward mode for gradient eval. */
      virtual MyReal runBWD(DataSet* data) = 0;

      /* Return network distribution to local processors */
      virtual void getGridDistribution(Config* config, 
                                       int* ilower_ptr, 
                                       int* iupper_ptr)=0;

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
      void getGridDistribution(Config* config,
                               int* ilower_ptr, 
                               int* iupper_ptr);
};

/**
 * Timestepping Solver
 */
class TimesteppingSolver : public Solver
{
   protected: 
      Network* network;
      int nchannels;

      MyReal *state; /* state at one layer for one example, dimension nchannels */

   public:
      /* Constructor */
      TimesteppingSolver(Config* config,
                         Network* network);

      /* Destructor */
      ~TimesteppingSolver();

      /* Run primal XBraid iterations */
      MyReal runFWD(DataSet* data);

      /* Run adjoint XBraid iterations */
      MyReal runBWD(DataSet* data);

      void getGridDistribution(Config* config,
                               int* ilower_ptr, 
                               int* iupper_ptr);
};