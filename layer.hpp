#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
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
      int nweights;                        /* Number of weights */
      int ndesign;                         /* Total number of design variables */

      int nconv;
      int csize;

      int     index;                       /* Number of the layer */
      double  dt;                          /* Step size for Layer update */
      double* weights;                     /* Weight matrix, flattened as a vector */
      double* weights_bar;                 /* Derivative of the Weight matrix*/
      double* bias;                        /* Bias */
      double* bias_bar;                    /* Derivative of bias */
      double  gamma_tik;                   /* Parameter for Tikhonov regularization of weights and bias */
      double  gamma_ddt;                   /* Parameter for DDT regularization of weights and bias */
      int     activ;                       /* Activaation function (enum element) */
      int     type;                        /* Type of the layer (enum element) */

      double *update;                      /* Auxilliary for computing fwd update */
      double *update_bar;                  /* Auxilliary for computing bwd update */

   public:
      /* Available activation functions */
      enum activation{TANH, RELU, SMRELU};  

      /* Available layer types */
      enum layertype{OPENZERO=0, OPENDENSE=1, DENSE=2, CLASSIFICATION=3, OPENCONV=4, OPENCONVMNIST=5, CONVOLUTION=6};

      Layer();
      Layer(int     idx,
            int     Type,
            int     dimI,
            int     dimO,
            int     dimB,
            double  deltaT,
            int     Activ,
            double  GammaTik,
            double  GammaDDT);

      Layer(int     idx,
            int     Type,
            int     dimI,
            int     dimO,
            int     dimB,
            int     dimW,   // number of weights
            double  deltaT,
            int     Activ,
            double  gammatik,
            double  gammaddt);

      Layer(int idx, 
            int Type,
            int dimI, 
            int dimO, 
            int dimB);

      virtual ~Layer();

      /* Set time step size */
      void setDt(double DT);

      /* Some Get..() functions */
      double getDt();
      double getGammaTik();
      double getGammaDDT();
      int    getActivation();
      int    getType();


      /* Get pointer to the weights bias*/
      double* getWeights();
      double* getBias();

      /* Get pointer to the weights bias bar */
      double* getWeightsBar();
      double* getBiasBar();

      /* Get the dimensions */
      int getDimIn();
      int getDimOut();
      int getDimBias();
      int getnDesign();

      int getnConv();
      int getCSize();

      /* Get the layer index (i.e. the time step) */
      int getIndex();

        /* Prints to screen */
      void print_data(double* data_Out);

      /* Activation function and derivative */
      double activation(double x);
      double dactivation(double x);


      /**
       * Pack weights and bias into a buffer 
       */
      void packDesign(double* buffer,
                      int     size);

      /**
       * Unpack weights and bias from a buffer 
       */
      void unpackDesign(double* buffer);


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
       * Regularization for the time-derivative of the layer weights
       */
      double evalRegulDDT(Layer* layer_prev,
                          double deltat);

      /**
       * Derivative of ddt-regularization term 
       */
      void evalRegulDDT_diff(Layer* layer_prev,
                             Layer* layer_next,
                             double deltat);


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
       * In:     compute_gradient - flag to determin if gradient should be computed (i.e. if weights_bar,bias_bar should be updated or not. In general, update is only done on the finest layer-grid.)
       */
      virtual void applyBWD(double* state,
                            double* state_bar,
                            int     compute_gradient) = 0;

      /**
       * On classification layer: applies the classification and evaluates loss/accuracy 
       */
      virtual void evalClassification(int      nexamples, 
                                      double** state,
                                      double** labels, 
                                      double*  loss_ptr, 
                                      double*  accuracy_ptr,
                                      int      output);

      /**
       * On classification layer: derivative of evalClassification 
       */
      virtual void evalClassification_diff(int      nexamples, 
                                          double** primalstate,
                                          double** adjointstate,
                                          double** labels, 
                                          int      compute_gradient);

      /* ReLu Activation and derivative */
      double ReLu_act(double x);
      double dReLu_act(double x);
        
      /* Smooth ReLu activation: Uses a quadratic approximation around zero (range: default 0.1) */
      double SmoothReLu_act(double x);
      double dSmoothReLu_act(double x);

      /* tanh Activation and derivative */
      double tanh_act(double x);
      double dtanh_act(double x);

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
                 int     activation,
                 double  gammatik, 
                 double  gammaddt);     
      ~DenseLayer();

      void applyFWD(double* state);

      void applyBWD(double* state,
                    double* state_bar,
                    int     compute_gradient);
};


/**
 * Opening Layer using dense weight matrix K \in R^{nxn}
 * Layer transformation: y = sigma(W*y_ex + b)  for examples y_ex \in \R^dimI
 */
class OpenDenseLayer : public DenseLayer {

  protected: 
      double* example;    /* Pointer to the current example data */

  public:
      OpenDenseLayer(int     dimI,
                     int     dimO,
                     int     activation,
                     double  gammatik);     
      ~OpenDenseLayer();

      void setExample(double* example_ptr);

      void applyFWD(double* state);

      void applyBWD(double* state,
                    double* state_bar,
                    int     compute_gradient);
};



/*
 * Opening layer that expands the data by zeros
 */
class OpenExpandZero : public Layer 
{
      protected: 
            double* example;    /* Pointer to the current example data */
      public:
            OpenExpandZero(int dimI,
                           int dimO);
            ~OpenExpandZero();

            void setExample(double* example_ptr);
           
            void applyFWD(double* state);
      
            void applyBWD(double* state,
                          double* state_bar,
                          int     compute_gradient);
};


/**
 * Classification layer
 */
class ClassificationLayer : public Layer
{
      protected: 
            double* probability;          /* vector of pedicted class probabilities */
            double* tmpstate;             /* temporarily holding the state */ 
            
      public:
            ClassificationLayer(int    idx,
                                int    dimI,
                                int    dimO,
                                double gammatik);
            ~ClassificationLayer();

            void applyFWD(double* state);
      
            void applyBWD(double* state,
                          double* state_bar,
                          int     compute_gradient);

            void evalClassification(int      nexamples, 
                                    double** state,
                                    double** labels, 
                                    double*  loss_ptr, 
                                    double*  accuracy_ptr,
                                    int      output);


            void evalClassification_diff(int      nexamples, 
                                         double** primalstate,
                                         double** adjointstate,
                                         double** labels, 
                                         int      compute_gradient);

            /**
             * Evaluate the cross entropy function 
             */
            double crossEntropy(double *finalstate,
                                double *label);

            /** 
             * Algorithmic derivative of evaluating cross entropy loss
             */
            void crossEntropy_diff(double *data_Out, 
                               double *data_Out_bar,
                               double *label,
                               double  loss_bar);

            /**
             * Compute the class probabilities
             * return 1 if predicted class was correct, 0 else.
             * out: *class_id_ptr holding the predicted class 
             */
            int prediction(double* data_out, 
                           double* label,
                           int*    class_id_ptr);

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


/**
 * Layer using a convolution C of size csize X csize, 
 * with nconv total convolutions. 
 * Layer transformation: y = y + dt * sigma(W(C) y + b)
 * if not openlayer: requires dimI = dimO !
 */
class ConvLayer : public Layer {

  public:
      ConvLayer(int     idx,
                int     dimI,
                int     dimO,
                int     csize_in,
                int     nconv_in,
                double  deltaT,
                int     Activ,
                double  Gammatik,
                double  Gammaddt);
      ~ConvLayer();

      void applyFWD(double* state);

      void applyBWD(double* state,
                    double* state_bar,
                    int     compute_gradient);

      double apply_conv(double* state,        // state vector to apply convolution to 
                      int     output_conv,    // output convolution
                      int     j,              // row index
                      int     k,              // column index
                      int     img_size_sqrt,  // sqrt of the image size
                      bool    transpose);     // apply the tranpose of the kernel

      /** 
       * This method is designed to be used only in the applyBWD. It computes the
       * derivative of the objective with respect to the weights. In particular
       * if you objective is $g$ and your kernel operator has value tau at index
       * a,b then
       *
       *   weights_bar[magic_index] = d_tau [ g] = \sum_{image j,k} tau state_{j+a,k+b} * update_bar_{j,k}
       *
       * Note that we assume that update_bar is 
       *
       *   update_bar = dt * dactivation * state_bar
       *
       * Where state_bar _must_ be at the old time. Note that the adjoint variable
       * state_bar carries withit all the information of the objective derivative.
       *
       * On exit this method modifies weights_bar
       */
      void updateWeightDerivative(
                      double* state,          // state vector
                      double * update_bar,    // combines derivative and adjoint info (see comments)
                      int     output_conv,    // output convolution
                      int     j,              // row index
                      int     k,              // column index
                      int     img_size_sqrt); // sqrt of the image size
};


/**
 * Opening Layer for use with convolutional layers.  Examples are replicated.
 * Layer transformation: y = ([I; I; ... I] y_ex)
 */
class OpenConvLayer : public Layer {

  protected: 
      double* example;    /* Pointer to the current example data */

  public:
      OpenConvLayer(int     dimI,
                    int     dimO);
      ~OpenConvLayer();

      void setExample(double* example_ptr);

      void applyFWD(double* state);

      void applyBWD(double* state,
                    double* state_bar,
                    int     compute_gradient);
};

/** 
 * Opening Layer for use with convolutional layers.  Examples are replicated
 * and then have an activation function applied.
 *
 * This layer is specially designed for MNIST
 *
 * Layer transformation: y = sigma([I; I; ... I] y_ex)
 */

class OpenConvLayerMNIST : public OpenConvLayer {

  public:
      OpenConvLayerMNIST(int     dimI,
                         int     dimO);
      ~OpenConvLayerMNIST();

      void applyFWD(double* state);

      void applyBWD(double* state,
                    double* state_bar,
                    int     compute_gradient);
};


