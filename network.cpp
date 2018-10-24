#include "network.hpp"

Network::Network()
{
   nlayers_global = 0;
   nlayers_local  = 0;
   nchannels      = 0;
   dt             = 0.0;
   loss           = 0.0;
   accuracy       = 0.0;
   gamma_tik      = 0.0;
   gamma_ddt      = 0.0;
   gamma_class    = 0.0;
   gradient       = NULL;
   design         = NULL;
   layers         = NULL;
   layer_left     = NULL;
   layer_right    = NULL;
}

Network::Network(int    nLayersGlobal,
                 int    nChannels, 
                 double deltaT,
                 double Gamma_tik, 
                 double Gamma_ddt,
                 double Gamma_class)
{

    /* Initilizize */
    nlayers_global   = nLayersGlobal;
    nchannels        = nChannels;
    dt               = deltaT;
    loss             = 0.0;
    accuracy         = 0.0;
    gamma_tik        = Gamma_tik;
    gamma_ddt        = Gamma_ddt;
    gamma_class      = Gamma_class;

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
                            double Weight_open_init)
{
    Layer* layer;
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
        else if (ilayer < nlayers_global-1) // Intermediate layer
        {
            layer = new DenseLayer(ilayer, nchannels, nchannels, dt, Activation, Gamma_tik);
        }
        else if (ilayer == nlayers_global-1) // Classification layer 
        {
            layer = new ClassificationLayer(ilayer, nchannels, nClasses, Gamma_class);
        }
        else
        {
            layer = NULL;
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


void Network::initialize(int    StartLayerID, 
                           int    EndLayerID, 
                           int    nFeatures,
                           int    nClasses,
                           int    Activation,
                           double Weight_init,
                           double Weight_open_init,
                           double Classification_init)
{

    startlayerID = StartLayerID;
    endlayerID   = EndLayerID;
    nlayers_local = endlayerID - startlayerID + 1;


    /* Sanity check */
    if (nFeatures > nchannels ||
        nClasses  > nchannels)
    {
        printf("ERROR! Choose a wider netword!\n");
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
        layers[storeID] = createLayer(ilayer, nFeatures, nClasses, Activation, gamma_tik, gamma_ddt, gamma_class, Weight_open_init);
        ndesign += layers[storeID]->getnDesign();
        
    }

    /* Allocate memory for network design and gradient variables */
    design   = new double[ndesign];
    gradient = new double[ndesign];

    /* Initialize  the layer weights and bias */
    int istart = 0;
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        int storeID = getLocalID(ilayer);
        if (ilayer == 0)  // Opening layer
        {
            layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), Weight_open_init);
            istart += layers[storeID]->getnDesign();
        }
        else if (ilayer < nlayers_global-1) // Intermediate layer
        {
            layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), Weight_init);
            istart += layers[storeID]->getnDesign();
        }
        else // Classification layer 
        {
            layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), Classification_init);
        }
    }

    /* Create and initialize left neighbouring layer, if exists */
    int leftID = startlayerID - 1;
    layer_left = createLayer(leftID, nFeatures, nClasses, Activation, gamma_tik, gamma_ddt, gamma_class, Weight_open_init);
    if (layer_left != NULL)
    {
        double *left_design   = new double[layer_left->getnDesign()];
        double *left_gradient = new double[layer_left->getnDesign()];
        layer_left->initialize(left_design, left_gradient, 0.0);
    }


    /* Create and initialize right neighbouring layer, if exists */
    int rightID = endlayerID + 1;
    layer_right = createLayer(rightID, nFeatures, nClasses, Activation, gamma_tik, gamma_ddt, gamma_class, Weight_open_init);
    if (layer_right != NULL)
    {
        double *right_design   = new double[layer_right->getnDesign()];
        double *right_gradient = new double[layer_right->getnDesign()];
        layer_right->initialize(right_design, right_gradient, 0.0);
    }

}    


double Network::evalRegulDDT(Layer* layer_old, 
                             Layer* layer_curr)
{
    double diff;
    double ddt = 0.0;

    /* Sanity check */
    if (layer_old->getDimIn()    != nchannels ||
        layer_old->getDimOut()   != nchannels ||
        layer_old->getDimBias()  != 1         ||
        layer_curr->getDimIn()   != nchannels ||
        layer_curr->getDimOut()  != nchannels ||
        layer_curr->getDimBias() != 1           )
        {
            printf("ERROR when evaluating ddt-regularization of intermediate Layers.\n"); 
            printf("Dimensions don't match. Check and change this routine.\n");
            exit(1);
        }

    for (int iw = 0; iw < nchannels * nchannels; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / dt;
        ddt += pow(diff,2);
    }
    diff = (layer_curr->getBias()[0] - layer_old->getBias()[0]) / dt;
    ddt += pow(diff,2);
    
    return gamma_ddt / 2.0 * ddt;
}                

void Network::evalRegulDDT_diff(Layer* layer_old, 
                                Layer* layer_curr,
                                double regul_bar)
{
    double diff;
    regul_bar = gamma_ddt * regul_bar;

    /* Derivative of the bias-term */
    diff = (layer_curr->getBias()[0] - layer_old->getBias()[0]) / pow(dt,2);
    layer_curr->getBiasBar()[0] += diff * regul_bar;
    layer_old->getBiasBar()[0]  -= diff * regul_bar;

    /* Derivative of the weights term */
    for (int iw = 0; iw < nchannels * nchannels; iw++)
    {
        diff = (layer_curr->getWeights()[iw] - layer_old->getWeights()[iw]) / pow(dt,2);
        layer_curr->getWeightsBar()[iw] += diff * regul_bar;
        layer_old->getWeightsBar()[iw]  -= diff * regul_bar;
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