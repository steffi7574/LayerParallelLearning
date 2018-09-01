#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "linalg.hpp"


#pragma once

/**
 * Abstract base class for the network layers 
 * Subclasses implement
 *    - applyFWD: Forward propagation of data 
 *    - applyBWD: Backward propagation of data 
 */
class Layer 
{
   protected:
      int dim_In;                          /* Dimension of incoming data */
      int dim_Out;                         /* Dimension of outgoing data */
      int dim_Bias;                        /* Dimension of the bias vector */
      int ndesign;                         /* Total number of design variables */

      int     index;                       /* Number of the layer */
      double  dt;                          /* Step size for Layer update */
      double* weights;                     /* Weight matrix, flattened as a vector */
      double* weights_bar;                 /* Derivative of the Weight matrix*/
      double* bias;                        /* Bias */
      double* bias_bar;                    /* Derivative of bias */
      double  (*activation)(double x);     /* the activation function */
      double  (*dactivation)(double x);    /* derivative of activation function */

      double *update;                      /* Auxilliary for computing fwd update */
      double *update_bar;                  /* Auxilliary for computing bwd update */

   public:
      Layer();
      Layer(int     idx,
            int     dimI,
            int     dimO,
            int     dimB,
            double  deltaT,
            double (*Activ)(double x),
            double (*dActiv)(double x));     
      virtual ~Layer();

      /* Set time step size */
      void setDt(double DT);

      /* Get a pointer to the weights bias*/
      double* getWeights();
      double* getBias();

      /* Get a pointer to the weights bias bar */
      double* getWeightsBar();
      double* getBiasBar();

      /* Get the dimensions */
      int getDimIn();
      int getDimOut();
      int getDimBias();
      int getnDesign();

        /* Prints to screen */
      void print_data(double* data_Out);

      /**
       * Initialize the layer primal and adjoint weights and biases
       * In: pointer to the global design and gradient vector memory 
       *     factor for scaling random initialization of primal variables
       */
      void initialize(double* design_ptr,
                      double* gradient_ptr,
                      double  factor);

      /**
       * Sets the bar variables to zero 
       */
      void resetBar();


      /**
       * Evaluate Tikhonov Regularization
       * Returns 1/2 * \|weights||^2 + 1/2 * \|bias\|^2
       */
      double evalTikh();

      /**
       * Derivative of Tikhonov Regularization
       */
      void evalTikh_diff(double regul_bar);

      /**
       * In opening layers: set pointer to the current example
       */
      virtual void setExample(double* example_ptr);

      /**
       * Forward propagation of an example 
       * In/Out: vector holding the current propagated example 
       */
      virtual void applyFWD(double* state) = 0;


      /**
       * Backward propagation of an example 
       * In:     data     - current example data
       * In/Out: data_bar - adjoint example data that is to be propagated backwards 
       */
      virtual void applyBWD(double* state,
                            double* state_bar) = 0;

      /**
       * Evaluates the loss function 
       */
      virtual double evalLoss(double *data_Out,
                              double *label);


      /** 
       * Derivative of evaluating the loss function 
       * In: data_Out, can be used to recompute intermediate states
       *     label for the current example 
       *     loss_bar: value holding the outer derivative
       * Out: data_Out_bar holding the derivative of loss wrt data_Out
       */
      virtual void evalLoss_diff(double *data_Out, 
                                 double *data_Out_bar,
                                 double *label,
                                 double  loss_bar);

      /**
       * Compute class probabilities,
       * return 1 if predicted class was correct, else 0.
       */
      virtual int prediction(double* data_Out,
                             double* label);

};

/**
 * Layer using square dense weight matrix K \in R^{nxn}
 * Layer transformation: y = y + dt * sigma(Wy + b)
 * if not openlayer: requires dimI = dimO !
 */
class DenseLayer : public Layer {

  public:
      DenseLayer(int     idx,
                 int     dimI,
                 int     dimO,
                 double  deltaT,
                 double (*Activ)(double x),
                 double (*dActiv)(double x));     
      ~DenseLayer();

      void applyFWD(double* state);

      void applyBWD(double* state,
                    double* state_bar);
};


/**
 * Opening Layer using dense weight matrix K \in R^{nxn}
 * Layer transformation: y = sigma(W*y_ex + b)  for examples y_ex \in \R^dimI
 */
class OpenDenseLayer : public DenseLayer {

  protected: 
      double* example;    /* Pointer to the current example data */

  public:
      OpenDenseLayer(int     idx,
                int     dimI,
                int     dimO,
                double  deltaT,
                double (*Activ)(double x),
                double (*dActiv)(double x));     
      ~OpenDenseLayer();

      void setExample(double* example_ptr);

      void applyFWD(double* state);

      void applyBWD(double* state,
                    double* state_bar);
};



/*
 * Opening layer that expands the data by zeros
 */
class OpenExpandZero : public Layer 
{
      protected: 
            double* example;    /* Pointer to the current example data */
      public:
            OpenExpandZero(int     dimI,
                           int     dimO,
                           double  deltaT);
            ~OpenExpandZero();

            void setExample(double* example_ptr);
           
            void applyFWD(double* state);
      
            void applyBWD(double* state,
                          double* state_bar);
};


/**
 * Classification layer
 */
class ClassificationLayer : public Layer
{
      protected: 
            double* probability;          /* vector of pedicted class probabilities */
      
      public:
            ClassificationLayer(int idx,
                                int dimI,
                                int dimO,
                                double  deltaT);
            ~ClassificationLayer();

            void applyFWD(double* state);
      
            void applyBWD(double* state,
                          double* state_bar);

            /**
             * Evaluate the loss function 
             */
            double evalLoss(double *finalstate,
                            double *label);

            /** 
             * Algorithmic derivative of evaluating loss
             */
            void evalLoss_diff(double *data_Out, 
                               double *data_Out_bar,
                               double *label,
                               double  loss_bar);

            /**
             * Compute the class probabilities
             * return 1 if predicted class was correct, 0 else.
             */
            int prediction(double* data_out, 
                           double* label);

            /**
             * Translate the data: 
             * Substracts the maximum value from all entries
             */
            void normalize(double* data);

            /**
             * Algorithmic derivative of the normalize funciton 
             */ 
            void normalize_diff(double* data, 
                                double* data_bar);
};