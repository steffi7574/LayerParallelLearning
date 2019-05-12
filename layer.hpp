#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "linalg.hpp"
#include "defs.hpp"
#include "config.hpp"


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
      MyReal  dt;                          /* Step size for Layer update */
      MyReal* weights;                     /* Weight matrix, flattened as a vector */
      MyReal* weights_bar;                 /* Derivative of the Weight matrix*/
      MyReal* bias;                        /* Bias */
      MyReal* bias_bar;                    /* Derivative of bias */
      MyReal  gamma_tik;                   /* Parameter for Tikhonov regularization of weights and bias */
      MyReal  gamma_ddt;                   /* Parameter for DDT regularization of weights and bias */
      int     activ;                       /* Activaation function (enum element) */
      int     type;                        /* Type of the layer (enum element) */

      MyReal *update;                      /* Auxilliary for computing fwd update */
      MyReal *update_bar;                  /* Auxilliary for computing bwd update */

   public:

      /* Available layer types */
      enum layertype{OPENZERO=0, OPENDENSE=1, DENSE=2, CLASSIFICATION=3, OPENCONV=4, OPENCONVMNIST=5, CONVOLUTION=6};

      Layer();
      Layer(int     idx,
            int     Type,
            int     dimI,
            int     dimO,
            int     dimB,
            int     dimW,   // number of weights
            MyReal  deltaT,
            int     Activ,
            MyReal  gammatik,
            MyReal  gammaddt);

      virtual ~Layer();

      /* Set time step size */
      void setDt(MyReal DT);

      /* Set design and gradient memory location */
      void setMemory(MyReal* design_memloc, 
                     MyReal* gradient_memloc);

      /* Some Get..() functions */
      MyReal getDt();
      MyReal getGammaTik();
      MyReal getGammaDDT();
      int    getActivation();
      int    getType();


      /* Get pointer to the weights bias*/
      MyReal* getWeights();
      MyReal* getBias();

      /* Get pointer to the weights bias bar */
      MyReal* getWeightsBar();
      MyReal* getBiasBar();

      /* Get the dimensions */
      int getDimIn();
      int getDimOut();
      int getDimBias();
      int getnWeights();
      int getnDesign();

      int getnConv();
      int getCSize();

      /* Get the layer index (i.e. the time step) */
      int getIndex();

        /* Prints to screen */
      void print_data(MyReal* data_Out);

      /* Activation function and derivative */
      MyReal activation(MyReal x);
      MyReal dactivation(MyReal x);


      /**
       * Pack weights and bias into a buffer 
       */
      void packDesign(MyReal* buffer,
                      int     size);

      /**
       * Unpack weights and bias from a buffer 
       */
      void unpackDesign(MyReal* buffer);


      /* Scales the weights by a factor and resets the gradient to zero. */
      void scaleDesign(MyReal  factor);

      /**
       * Sets the bar variables to zero 
       */
      void resetBar();

      /**
       * Evaluate Tikhonov Regularization
       * Returns 1/2 * \|weights||^2 + 1/2 * \|bias\|^2
       */
      MyReal evalTikh();

      /**
       * Derivative of Tikhonov Regularization
       */
      void evalTikh_diff(MyReal regul_bar);

     
      /**
       * Regularization for the time-derivative of the layer weights
       */
      MyReal evalRegulDDT(Layer* layer_prev,
                          MyReal deltat);

      /**
       * Derivative of ddt-regularization term 
       */
      void evalRegulDDT_diff(Layer* layer_prev,
                             Layer* layer_next,
                             MyReal deltat);


      /**
       * In opening layers: set pointer to the current example
       */
      virtual void setExample(MyReal* example_ptr);

      /**
       * In classification layers: set pointer to the current label 
       */
      virtual void setLabel(MyReal* label_ptr);

      /**
       * Forward propagation of an example 
       * In/Out: vector holding the current propagated example 
       */
      virtual void applyFWD(MyReal* state) = 0;


      /**
       * Backward propagation of an example 
       * In:     data     - current example data
       * In/Out: data_bar - adjoint example data that is to be propagated backwards 
       * In:     compute_gradient - flag to determin if gradient should be computed (i.e. if weights_bar,bias_bar should be updated or not. In general, update is only done on the finest layer-grid.)
       */
      virtual void applyBWD(MyReal* state,
                            MyReal* state_bar,
                            int     compute_gradient) = 0;



      virtual MyReal crossEntropy(MyReal *finalstate);

      virtual int prediction(MyReal* data_out, 
                             int*    class_id_ptr);

      /* ReLu Activation and derivative */
      MyReal ReLu_act(MyReal x);
      MyReal dReLu_act(MyReal x);
        
      /* Smooth ReLu activation: Uses a quadratic approximation around zero (range: default 0.1) */
      MyReal SmoothReLu_act(MyReal x);
      MyReal dSmoothReLu_act(MyReal x);

      /* tanh Activation and derivative */
      MyReal tanh_act(MyReal x);
      MyReal dtanh_act(MyReal x);

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
                 MyReal  deltaT,
                 int     activation,
                 MyReal  gammatik, 
                 MyReal  gammaddt);     
      ~DenseLayer();

      void applyFWD(MyReal* state);

      void applyBWD(MyReal* state,
                    MyReal* state_bar,
                    int     compute_gradient);
};


/**
 * Opening Layer using dense weight matrix K \in R^{nxn}
 * Layer transformation: y = sigma(W*y_ex + b)  for examples y_ex \in \R^dimI
 */
class OpenDenseLayer : public DenseLayer {

  protected: 
      MyReal* example;    /* Pointer to the current example data */

  public:
      OpenDenseLayer(int     dimI,
                     int     dimO,
                     int     activation,
                     MyReal  gammatik);     
      ~OpenDenseLayer();

      void setExample(MyReal* example_ptr);

      void applyFWD(MyReal* state);

      void applyBWD(MyReal* state,
                    MyReal* state_bar,
                    int     compute_gradient);
};



/*
 * Opening layer that expands the data by zeros
 */
class OpenExpandZero : public Layer 
{
      protected: 
            MyReal* example;    /* Pointer to the current example data */
      public:
            OpenExpandZero(int dimI,
                           int dimO);
            ~OpenExpandZero();

            void setExample(MyReal* example_ptr);
           
            void applyFWD(MyReal* state);
      
            void applyBWD(MyReal* state,
                          MyReal* state_bar,
                          int     compute_gradient);
};


/**
 * Classification layer
 */
class ClassificationLayer : public Layer
{
      protected: 
            MyReal* label;                /* Pointer to the current label vector */

            MyReal* probability;          /* vector of pedicted class probabilities */
            
      public:
            ClassificationLayer(int    idx,
                                int    dimI,
                                int    dimO,
                                MyReal gammatik);
            ~ClassificationLayer();

            void setLabel(MyReal* label_ptr);

            void applyFWD(MyReal* state);
      
            void applyBWD(MyReal* state,
                          MyReal* state_bar,
                          int     compute_gradient);

            /**
             * Evaluate the cross entropy function 
             */
            MyReal crossEntropy(MyReal *finalstate);

            /** 
             * Algorithmic derivative of evaluating cross entropy loss
             */
            void crossEntropy_diff(MyReal *data_Out, 
                                   MyReal *data_Out_bar,
                                   MyReal  loss_bar);

            /**
             * Compute the class probabilities
             * return 1 if predicted class was correct, 0 else.
             * out: *class_id_ptr holding the predicted class 
             */
            int prediction(MyReal* data_out, 
                           int*    class_id_ptr);

            /**
             * Translate the data: 
             * Substracts the maximum value from all entries
             */
            void normalize(MyReal* data);

            /**
             * Algorithmic derivative of the normalize funciton 
             */ 
            void normalize_diff(MyReal* data, 
                                MyReal* data_bar);

};


/**
 * Layer using a convolution C of size csize X csize, 
 * with nconv total convolutions. 
 * Layer transformation: y = y + dt * sigma(W(C) y + b)
 * if not openlayer: requires dimI = dimO !
 */
class ConvLayer : public Layer {

     int csize2;
     int fcsize;

     int img_size;
     int img_size_sqrt;

  public:
      ConvLayer(int     idx,
                int     dimI,
                int     dimO,
                int     csize_in,
                int     nconv_in,
                MyReal  deltaT,
                int     Activ,
                MyReal  Gammatik,
                MyReal  Gammaddt);
      ~ConvLayer();

      void applyFWD(MyReal* state);

      void applyBWD(MyReal* state,
                    MyReal* state_bar,
                    int     compute_gradient);

      inline MyReal apply_conv(MyReal* state,        // state vector to apply convolution to 
                      int     output_conv,    // output convolution
                      int     j,              // row index
                      int     k);             // column index

      inline MyReal apply_conv_trans(MyReal* state,        // state vector to apply convolution to 
                      int     output_conv,    // output convolution
                      int     j,              // row index
                      int     k);             // column index

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
      inline MyReal updateWeightDerivative(
                      MyReal* state,          // state vector
                      MyReal * update_bar,    // combines derivative and adjoint info (see comments)
                      int     output_conv,    // output convolution
                      int     j,              // row index
                      int     k);             // column index
};


/**
 * Opening Layer for use with convolutional layers.  Examples are replicated.
 * Layer transformation: y = ([I; I; ... I] y_ex)
 */
class OpenConvLayer : public Layer {

  protected: 
      MyReal* example;    /* Pointer to the current example data */

  public:
      OpenConvLayer(int     dimI,
                    int     dimO);
      ~OpenConvLayer();

      void setExample(MyReal* example_ptr);

      void applyFWD(MyReal* state);

      void applyBWD(MyReal* state,
                    MyReal* state_bar,
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

      void applyFWD(MyReal* state);

      void applyBWD(MyReal* state,
                    MyReal* state_bar,
                    int     compute_gradient);
};


