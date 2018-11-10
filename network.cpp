#include "network.hpp"
#include<assert.h>

Network::Network()
{
   nlayers_global = 0;
   nlayers_local  = 0;
   nchannels      = 0;
   dt             = 0.0;
   loss           = 0.0;
   accuracy       = 0.0;
   gradient       = NULL;
   design         = NULL;
   layers         = NULL;
   layer_left     = NULL;
   layer_right    = NULL;
}

Network::Network(int    nLayersGlobal,
                 int    StartLayerID, 
                 int    EndLayerID, 
                 int    nFeatures,
                 int    nClasses,
                 int    nChannels, 
                 int    Activation,
                 double deltaT,
                 double gamma_tik, 
                 double gamma_ddt, 
                 double gamma_class,
                 double Weight_open_init,
                 int    networkType,
                 int    type_openlayer)
{
    /* Initilizize */
    nlayers_global   = nLayersGlobal;
    startlayerID     = StartLayerID;
    endlayerID       = EndLayerID;
    nlayers_local    = endlayerID - startlayerID + 1;

    nchannels        = nChannels;
    dt               = deltaT;
    loss             = 0.0;
    accuracy         = 0.0;


    /* Sanity check */
    if (nFeatures > nchannels ||
        nClasses  > nchannels)
    {
        printf("ERROR! Choose a wider netword!\n");
        printf(" -- nFeatures = %d\n", nFeatures);
        printf(" -- nChannels = %d\n", nChannels);
        printf(" -- nClasses = %d\n", nClasses);
        exit(1);
    }

    printf("creating layers startid %d endid %d, nlayer_local %d\n", startlayerID, endlayerID, nlayers_local);

   /* --- Create the layers --- */
    layers  = new Layer*[nlayers_local];
    ndesign = 0;
 
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        /* Create a layer at time step ilayer. Local storage at ilayer - startlayerID */
        int storeID = getLocalID(ilayer);
        layers[storeID] = createLayer(ilayer, nFeatures, nClasses, Activation, gamma_tik, gamma_ddt, gamma_class, Weight_open_init, networkType, type_openlayer);
        ndesign += layers[storeID]->getnDesign();
        
    }

    /* Allocate memory for network design and gradient variables */
    design   = new double[ndesign];
    gradient = new double[ndesign];

    /* Create left neighbouring layer */
    int leftID = startlayerID - 1;
    layer_left = createLayer(leftID, nFeatures, nClasses, Activation, gamma_tik, gamma_ddt, gamma_class, Weight_open_init, networkType, type_openlayer);

    /* Create right neighbrouing layer */
    int rightID = endlayerID + 1;
    layer_right = createLayer(rightID, nFeatures, nClasses, Activation, gamma_tik, gamma_ddt, gamma_class, Weight_open_init, networkType, type_openlayer);
}             

  



Network::~Network()
{
    /* Delete the layers */
    for (int ilayer = 0; ilayer < nlayers_local; ilayer++)
    {
        delete layers[ilayer];
    }
    delete [] layers;
    delete [] design;
    delete [] gradient;

    if (layer_left != NULL)
    {
        delete [] layer_left->getWeights();
        delete [] layer_left->getWeightsBar();
        delete layer_left;
    }

    if (layer_right != NULL)
    {
        delete [] layer_right->getWeights();
        delete [] layer_right->getWeightsBar();
        delete layer_right;
    }
}

int Network::getnChannels() { return nchannels; }

int Network::getnLayers() { return nlayers_global; }

double Network::getDT() { return dt; }

int Network::getLocalID(int ilayer) 
{
    int idx = ilayer - startlayerID;
    return idx;
}

double Network::getLoss() { return loss; }

double Network::getAccuracy() { return accuracy; }

int Network::getnDesign() { return ndesign; }

double* Network::getDesign() { return design; }
       
double* Network::getGradient() { return gradient; }


Layer* Network::createLayer(int    ilayer, 
                            int    nFeatures,
                            int    nClasses,
                            int    Activation,
                            double Gamma_tik,
                            double Gamma_ddt,
                            double Gamma_class,
                            double Weight_open_init,
                            int    networkType,
                            int    type_openlayer)
{
    Layer* layer;
    if(networkType==DENSE) {
        if (ilayer == 0)  // Opening layer
        {
            if (Weight_open_init == 0.0)
            {
                layer  = new OpenExpandZero(nFeatures, nchannels);
            }
            else
            {
                layer = new OpenDenseLayer(nFeatures, nchannels, Activation, Gamma_tik);
            }
        }
        else if (0 < ilayer && ilayer < nlayers_global-1) // Intermediate layer
        {
            layer = new DenseLayer(ilayer, nchannels, nchannels, dt, Activation, Gamma_tik, Gamma_ddt);
        }
        else if (ilayer == nlayers_global-1) // Classification layer 
        {
            layer = new ClassificationLayer(ilayer, nchannels, nClasses, Gamma_class);
        }
        else
        {
            layer = NULL;
        }
    }
    else if(networkType==CONVOLUTIONAL) {
        if (ilayer == 0)  // Opening layer
        {
            /**< (Weight_open_init == 0.0) not needed for convolutional layers*/
            if (type_openlayer == 0)
            {
                layer = new OpenConvLayer(nFeatures, nchannels);
            }
            else if (type_openlayer == 1)
            {
                layer = new OpenConvLayerMNIST(nFeatures, nchannels);
            }
        }
        else if (0 < ilayer && ilayer < nlayers_global-1) // Intermediate layer
        {
            int convolution_size = 3;
            layer = new ConvLayer(ilayer, nchannels, nchannels,
                                            convolution_size,nchannels/nFeatures, 
                                            dt, Activation, Gamma_tik);
        }
        else if (ilayer == nlayers_global-1) // Classification layer 
        {
            layer = new ClassificationLayer(ilayer, nchannels, nClasses, Gamma_class);
        }
        else
        {
            layer = NULL;
        }
    }

    return layer;
}                        

Layer* Network::getLayer(int layerindex)
{
    Layer* layer;

    if (layerindex == startlayerID - 1)
    {
        layer = layer_left;
    } 
    else if (startlayerID <= layerindex && layerindex <= endlayerID)
    {
        layer = layers[getLocalID(layerindex)];
    }
    else if (layerindex == endlayerID + 1)
    {
        layer = layer_right;
    }
    else
    {
        layer = NULL;
    }

    return layer;
}


void Network::initialize(double Weight_open_init,
                         double Weight_init,
                         double Classification_init)
{
    double factor;

    /* Initialize  the layer weights and bias */
    int istart = 0;
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        if (ilayer == 0)  // Opening layer
        {
            factor = Weight_open_init;
        }
        else if (0 < ilayer && ilayer < nlayers_global-1) // Intermediate layer
        {
            factor = Weight_init;
        }
        else // Classification layer 
        {
            factor = Classification_init;
        }
        int storeID = getLocalID(ilayer);
        layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), factor);
        istart += layers[storeID]->getnDesign();
    }

    /* Create and initialize left neighbouring layer, if exists */
    if (layer_left != NULL)
    {
        double *left_design   = new double[layer_left->getnDesign()];
        double *left_gradient = new double[layer_left->getnDesign()];
        layer_left->initialize(left_design, left_gradient, 0.0);
    }

    /* Create and initialize right neighbouring layer, if exists */
    if (layer_right != NULL)
    {
        double *right_design   = new double[layer_right->getnDesign()];
        double *right_gradient = new double[layer_right->getnDesign()];
        layer_right->initialize(right_design, right_gradient, 0.0);
    }
}    


void Network::MPI_CommunicateNeighbours(MPI_Comm comm)
{
    int myid, comm_size;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &comm_size);
    MPI_Request sendlastreq, recvlastreq;
    MPI_Request sendfirstreq, recvfirstreq;
    MPI_Status status;

    /* Allocate buffers */
    int size = (nchannels*nchannels+nchannels);
    double* sendlast  = new double[size];
    double* recvlast  = new double[size];
    double* sendfirst = new double[size];
    double* recvfirst = new double[size];

    /* --- All but the first process receive the last layer from left neighbour --- */
    if (myid > 0)
    {
        /* Receive from left neighbour */
        int source = myid - 1;
        MPI_Irecv(recvlast, size, MPI_DOUBLE, source, 0, comm, &recvlastreq);
    }

    /* --- All but the last process sent their last layer to right neighbour --- */
    if (myid < comm_size-1)
    {
        /* Pack the last layer into a buffer */
        layers[getLocalID(endlayerID)]->packDesign(sendlast, size);

       /* Send to right neighbour */
        int receiver = myid + 1;
        MPI_Isend(sendlast, size, MPI_DOUBLE, receiver, 0, comm, &sendlastreq);
    }

    /* --- All but the last processor recv the first layer from the right neighbour --- */
    if (myid < comm_size - 1)
    {
        /* Receive from right neighbour */
        int source = myid + 1;
        MPI_Irecv(recvfirst, size, MPI_DOUBLE, source, 1, comm, &recvfirstreq);
    }

    /* --- All but the first processor send their first layer to the left neighbour --- */
    if (myid > 0)
    {
        /* Pack the first layer into a buffer */
        layers[getLocalID(startlayerID)]->packDesign(sendfirst, size);

        /* Send to left neighbour */
        int receiver = myid - 1;
        MPI_Isend(sendfirst, size, MPI_DOUBLE, receiver, 1, comm, &sendfirstreq);
    }

    /* Wait to finish up communication */
    if (myid > 0)              MPI_Wait(&recvlastreq, &status);
    if (myid < comm_size - 1)  MPI_Wait(&sendlastreq, &status);
    if (myid < comm_size - 1)  MPI_Wait(&recvfirstreq, &status);
    if (myid > 0)              MPI_Wait(&sendfirstreq, &status);

    /* Unpack and store the left received layer */
    if (myid > 0)
    {
        layer_left->unpackDesign(recvlast);
    }

    /* Unpack and store the right received layer */
    if (myid < comm_size - 1)
    {
        layer_right->unpackDesign(recvfirst);
    }

    /* Free the buffer */
    delete [] sendlast;
    delete [] recvlast;
    delete [] sendfirst;
    delete [] recvfirst;

}
