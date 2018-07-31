#include <stdlib.h>
#include <stdio.h>
#include "lib.h"


int main()
{

    RealReverse *Ytrain;      /**< State variables: contains the training data from Lars*/
    RealReverse *theta;      /**< theta variables for the network */
    RealReverse  objective;   /**< Objective function to be minimized */
    double      *Ytarget;     /**< Contains the target values from Lars */
    int         *batch;       /**< Contains indicees of the batch elements */
    int          nexamples;   /**< Number of elements in the training data */
    int          nbatch;      /**< Size of a batch */
    int          nstate;      /**< dimension of the training data */
    int          ntheta;     /**< dimension of the theta variables */
    int          nlayers;      /**< Number of layers / time steps */
    int          nchannels;   /**< Number of channels of the netword (width) */
    double       deltat;      /**< Time step size */
    double       T;           /**< Final time */
    double       theta0;      /**< Initial design value */
    double       alpha;       /**< Regularization parameter */
    double      *gradient;    /**< Gradient of objective function wrt theta */
    double       gnorm;       /**< Norm of the gradient */
    

    /* Problem setup */
    nexamples = 5000;
    nchannels = 4;
    nlayers    = 32;
    T         = 10.0;
    theta0    = 1e-2;
    alpha     = 1e-2;

    nbatch  = nexamples;
    deltat  = T/nlayers;
    nstate  = nchannels * nexamples; 
    ntheta = (nchannels * nchannels + 1 )* nlayers;

    /* Allocate memory */
    Ytrain   = (RealReverse*) malloc(nstate*sizeof(RealReverse));
    theta   = (RealReverse*) malloc(ntheta*sizeof(RealReverse));
    gradient = (double*) malloc(ntheta*sizeof(double));
    Ytarget  = (double*) malloc(nstate*sizeof(double));
    batch    = (int*) malloc(nbatch*sizeof(int));


    /* Read in data */
    read_data ("Ytrain.transpose.dat", Ytrain, nstate);
    read_data ("Ytarget.transpose.dat", Ytarget, nstate);

    /* Initialize design */
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        theta[itheta] = theta0; 
    }

    /* Initialize the batch (same as examples for now) */
    for (int ibatch = 0; ibatch < nbatch; ibatch++)
    {
        batch[ibatch] = ibatch;
    }

    /* Set up CoDiPack */
    RealReverse::TapeType& codiTape = RealReverse::getGlobalTape();
    codiTape.setActive();
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        codiTape.registerInput( theta[itheta] );
    }


    /* Time-loop */
    objective = 0.0;
    for (int ts = 0; ts <= nlayers; ts++)
    {
        /* Compute regularization term */
        objective += alpha * regularization_theta(theta, ts, deltat, nlayers, nchannels);

        /* Move to next layer */
        take_step(Ytrain, theta, ts, deltat, batch, nbatch, nchannels, 0);

        /* If last layer: Compute loss */
        if ( ts == nlayers )
        {
            RealReverse tmp = 1./ nbatch * loss(Ytrain, Ytarget, batch, nbatch, nchannels);
            objective += tmp;
        }

    }

    /* Get the derivative */
    codiTape.setPassive();
    objective.setGradient(1.0);
    codiTape.evaluate();
    for (int itheta = 0; itheta < ntheta; itheta++)
    {
        gradient[itheta] = theta[itheta].getGradient(); 
        gnorm += pow(gradient[itheta],2);
    }
    gnorm = sqrt(gnorm);

    /* output */
    printf("\n Loss:         %1.14e", getValue(objective));
    printf("\n Gradientnorm: %1.14e", gnorm);
    printf("\n\n");

    /* Write data to file */
    write_data("Yout.dat", Ytrain, nstate);
    write_data("gradient.dat", gradient, ntheta);

    /* free memory */
    free(Ytrain);
    free(Ytarget);
    free(theta);
    free(batch);

    return 0;
}