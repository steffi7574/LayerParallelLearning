#include <stdlib.h>
#include <stdio.h>
#include "lib.h"


int main()
{

    RealReverse *Ytrain;       /**< State variables: contains the training data from Lars*/
    RealReverse *design;       /**< Design variables for the network */
    RealReverse  objective;    /**< Objective function to be minimized */
    double      *Ytarget;      /**< Contains the target values from Lars */
    int         *batch;        /**< Contains indicees of the batch elements */
    int          nexamples;    /**< Number of elements in the training data */
    int          nbatch;       /**< Size of a batch */
    int          nstate;       /**< dimension of the training data */
    int          ndesign;      /**< dimension of the design variables */
    int          ntimes;       /**< Number of layers / time steps */
    int          nchannels;    /**< Number of channels of the netword (width) */
    double       deltat;       /**< Time step size */
    double       T;            /**< Final time */
    double       theta0;       /**< Initial design value */
    double       alpha;        /**< Regularization parameter */
    

    /* Problem setup */
    nexamples = 5000;
    nchannels = 4;
    ntimes    = 32;
    T         = 10.0;
    theta0    = 1e-2;
    alpha     = 1e-2;

    nbatch  = nexamples;
    deltat  = T/ntimes;
    nstate  = nchannels * nexamples; 
    ndesign = (nchannels * nchannels + 1 )* ntimes;

    /* Allocate memory */
    Ytrain  = (RealReverse*) malloc(nstate*sizeof(RealReverse));
    design  = (RealReverse*) malloc(ndesign*sizeof(RealReverse));
    Ytarget = (double*) malloc(nstate*sizeof(double));
    batch   = (int*) malloc(nbatch*sizeof(int));


    /* Read in data */
    read_data ("Ytrain.transpose.dat", Ytrain, nstate);
    read_data ("Ytarget.transpose.dat", Ytarget, nstate);

    /* Initialize design */
    for (int idesign = 0; idesign < ndesign; idesign++)
    {
        design[idesign] = theta0; 
    }

    /* Initialize the batch (same as examples for now) */
    for (int ibatch = 0; ibatch < nbatch; ibatch++)
    {
        batch[ibatch] = ibatch;
    }

    /* Time-loop */
    objective = 0.0;
    for (int ts = 0; ts <= ntimes; ts++)
    {
        /* Compute regularization term */
        objective += alpha * regularization(design, ts, deltat, ntimes, nchannels);

        /* Move to next layer */
        take_step(Ytrain, design, ts, deltat, batch, nbatch, nchannels, 0);

        /* If last layer: Compute loss */
        if ( ts == ntimes )
        {
            RealReverse tmp = 1./ nbatch * loss(Ytrain, Ytarget, batch, nbatch, nchannels);
            objective += tmp;
        }

    }

    /* output */
    write_data ("Yout.dat", Ytrain, nstate);

    // printf("Y %1.14e\n", Ytrain[99 * nchannels + 0]);
    // printf("Y %1.14e\n", Ytrain[99 * nchannels + 1]);
    // printf("Y %1.14e\n", Ytrain[99 * nchannels + 2]);
    // printf("Y %1.14e\n", Ytrain[99 * nchannels + 3]);

    printf("\n Loss: %1.14e\n\n", objective.getValue());

    /* free memory */
    free(Ytrain);
    free(Ytarget);
    free(design);
    free(batch);

    return 0;
}