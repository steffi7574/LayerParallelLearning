#ifndef BRAID_WRAPPER_HPP_INCLUDED
#define BRAID_WRAPPER_HPP_INCLUDED


#include "braid.h"
#include "Layer.hpp"

/* Define the app structure */
typedef struct _braid_App_struct
{
    int      myid;             /* Processor rank*/
    double  *Ytrain;           /* Training data points of matlab's peak() function */
    double  *Ctrain;           /* Training data: Label vectors (C) */
    double  *Yval;             /* Validation data points of matlab's peak() function */
    double  *Cval;             /* Validation data: Label vectors (C) */
    double  *theta;            /* theta variables */
    double  *theta_grad;       /* Gradient of objective function wrt theta */
    double  *theta_open;       /* Weights and bias of the opening layer */
    double  *theta_open_grad;  /* Gradient of the weights and bias of the opening layer */
    double  *classW;           /* Weights of the classification problem (W) */
    double  *classW_grad;      /* Gradient wrt the classification weights */
    double  *classMu;          /* Bias of the classification problem (mu) */
    double  *classMu_grad;     /* Gradient wrt the classification bias */
    int      ntraining;        /* Elements of the training data */
    int      nvalidation;      /* Elements of the validation data */
    int      nfeatures;        /* Number of features in the training data */
    int      nclasses;         /* Number of classes */
    int      nchannels;        /* Width of the network */
    int      ntheta_open;      /* Number of weights in opening layer (if zero, just expand the data with zeros) */
    int      nlayers;          /* number of time-steps / layers */
    Layer    *layer;           /* General layer architecture */
    Layer    *openlayer;       /* Opening Layer architecture */
    double   gamma_theta_tik;  /* Relaxation parameter for theta tikhonov */
    double   gamma_theta_ddt;  /* Relaxation parameter for theta time-derivative */
    double   gamma_class;      /* Relaxation parameter for the classification weights W and bias mu */
    double   deltaT;           /* Time-step size on fine grid */
    double   accuracy;         /* accur_train of the training data */
    double   theta_regul;      /* Theta-Regularization term of the objective function */
    double   class_regul;      /* Classifier-Regularization term of the objective function */
    double   loss;             /* Loss term of the objective function */
    int      training;         /* Flag, if training (1) or not (0) */
    int      output;           /* Determine, if loss function writes to prediction.dat */
} my_App;


/* Define the state vector at one time-step */
typedef struct _braid_Vector_struct
{
   double *Y;            /* Network state at one layer */

} my_Vector;


int 
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status);

int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr);

int
my_Init_diff(braid_App     app,
             double        t,
             braid_Vector  ubar);

int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr);


int
my_Free(braid_App    app,
        braid_Vector u);


int
my_Sum(braid_App     app,
       double        alpha,
       braid_Vector  x,
       double        beta,
       braid_Vector  y);


int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr);


int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus);


int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus);


int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus);


int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus);


int 
my_ObjectiveT(braid_App              app,
              braid_Vector           u,
              braid_ObjectiveStatus  ostatus,
              double                *objective_ptr);


int
my_ObjectiveT_diff(braid_App            app,
                  braid_Vector          u,
                  braid_Vector          u_bar,
                  braid_Real            f_bar,
                  braid_ObjectiveStatus ostatus);

int
my_Step_diff(braid_App         app,
             braid_Vector      ustop,     /**< input, u vector at *tstop* */
             braid_Vector      u,         /**< input, u vector at *tstart* */
             braid_Vector      ustop_bar, /**< input / output, adjoint vector for ustop */
             braid_Vector      u_bar,     /**< input / output, adjoint vector for u */
             braid_StepStatus  status);

int 
my_ResetGradient(braid_App app);



#endif // BRAID_WRAPPER_HPP_INCLUDED