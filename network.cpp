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
}

int Network::getnChannels() { return nchannels; }

int Network::getnLayers() { return nlayers_global; }

double Network::getDT() { return dt; }

int Network::getLocalID(int ilayer) 
{
    int idx = ilayer - startlayerID;
    if (idx < 0 || idx > nlayers_global-1) 
    {
           printf("\n\nERROR! Something went wrong with local storage of layers! \n");
           printf("ilayer %d, startlayerID %d\n\n", ilayer, startlayerID);
    }

    return idx;
}

double Network::getLoss() { return loss; }

double Network::getAccuracy() { return accuracy; }

int Network::getnDesign() { return ndesign; }

double* Network::getDesign() { return design; }
       
double* Network::getGradient() { return gradient; }





void Network::createLayers(int    StartLayerID, 
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
        if (ilayer == 0)  // Opening layer
        {
            if (Weight_open_init == 0.0)
            {
               layers[storeID]  = new OpenExpandZero(nFeatures, nchannels);
            //    printf("Creating OpenExpandZero-Layer at %d local %d\n", ilayer, storeID);
            }
            else
            {
               layers[storeID] = new OpenDenseLayer(nFeatures, nchannels, Activation, gamma_tik);
            //    printf("Creating OpenDense-Layer at %d local %d\n", layers[storeID]->getIndex(), storeID);
            }
            ndesign += layers[storeID]->getnDesign();
        }
        else if (ilayer < nlayers_global-1) // Intermediate layer
        {
            layers[storeID] = new DenseLayer(ilayer, nchannels, nchannels, dt, Activation, gamma_tik);
            ndesign += layers[storeID]->getnDesign();
            // printf("Creating Dense-Layer at %d local %d\n", layers[storeID]->getIndex(), storeID);
        }
        else // Classification layer 
        {
            layers[storeID] = new ClassificationLayer(nlayers_global-1, nchannels, nClasses, gamma_class);
            ndesign += layers[storeID]->getnDesign();
            // printf("Creating Classification-Layer at %d local %d\n", layers[storeID]->getIndex(), storeID);
        }
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
}    





// void Network::applyFWD(int      nexamples,
//                        double **examples,
//                        double **labels)
// {
//     int success;
//     double* state = new double[nchannels];

//     /* Propagate the examples */
//     loss    = 0.0;
//     success = 0;
//     for (int iex = 0; iex < nexamples; iex++)
//     { 
//         /* Load input data */
//         layers[0]->setExample(examples[iex]);
       
//         /* Propagate through all layers */ 
//         for (int ilayer = 0; ilayer < nlayers; ilayer++)
//         {
//             /* Apply the next layer */
//             layers[ilayer]->applyFWD(state);
//         }

//         /* Evaluate loss */
//         loss += layers[nlayers-1]->evalLoss(state, labels[iex]);

//         /* Test for successful prediction */
//         success += layers[nlayers-1]->prediction(state, labels[iex]);
//     }
        
//     /* Set loss and accuracy */
//     loss     = 1. / nexamples * loss;
//     accuracy = 100.0 * (double) success / nexamples;

//     delete [] state;
// }


// double Network::evalRegularization()
// {
//     double regul_tikh  = 0.0;
//     double regul_ddt   = 0.0;

//     /* Evaluate regularization terms for each layer */
//     for (int ilayer = 0; ilayer < nlayers; ilayer++)
//     {
//         regul_tikh += layers[ilayer]->evalTikh();
//         if (ilayer > 1 && ilayer < nlayers - 1) regul_ddt += evalRegulDDT(layers[ilayer-1], layers[ilayer]);
//     }

//     return regul_tikh + regul_ddt;
// }


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



Layer* Network::MPI_CommunicateLayerNeighbours(MPI_Comm comm)
{
    int myid, comm_size;
    int idx;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Status status;

    // int nchannels    = getnChannels();
    int size         = (8 + 2*(nchannels*nchannels+nchannels));

    Layer* recvlayer = NULL;

    /* Allocate a buffer */
    double* sendbuffer = new double[size];
    double* recvbuffer = new double[size];


    /* All but the last processor send their last layer to the next neighbour on their right */
    if (myid < comm_size-1)
    {
        int lastlayerIDX = getLocalID(endlayerID);
        Layer* sendlayer = layers[lastlayerIDX];
        int nweights     = sendlayer->getnDesign() - sendlayer->getDimBias();
        int nbias        = sendlayer->getnDesign() - sendlayer->getDimIn() * sendlayer->getDimOut();

        /* Pack the layer into a buffer */

        idx = 0;
        sendbuffer[idx] = sendlayer->getType();       idx++;
        sendbuffer[idx] = sendlayer->getIndex();      idx++;
        sendbuffer[idx] = sendlayer->getDimIn();      idx++;
        sendbuffer[idx] = sendlayer->getDimOut();     idx++;
        sendbuffer[idx] = sendlayer->getDimBias();    idx++;
        sendbuffer[idx] = sendlayer->getActivation(); idx++;
        sendbuffer[idx] = sendlayer->getnDesign();    idx++;
        sendbuffer[idx] = sendlayer->getGamma();      idx++;
        for (int i = 0; i < nweights; i++)
        {
            sendbuffer[idx] = sendlayer->getWeights()[i];     idx++;
            sendbuffer[idx] = sendlayer->getWeightsBar()[i];  idx++;
        }
        for (int i = 0; i < nbias; i++)
        {
            sendbuffer[idx] = sendlayer->getBias()[i];     idx++;
            sendbuffer[idx] = sendlayer->getBiasBar()[i];  idx++;
        }
        /* Set the rest to zero */
        for (int i = idx; i < size; i++)
        {
            sendbuffer[idx] = 0.0;  idx++;
        }

        /* Send the buffer */
        int receiver = myid + 1;
        MPI_Send(sendbuffer, size, MPI_DOUBLE, receiver, 0, comm);
    }

    
    /* All but the first processor receive a layer */
    if (myid > 0)
    {
        /* Receive the buffer */
        int sender = myid - 1;
        MPI_Recv(recvbuffer, size, MPI_DOUBLE, sender, 0, comm, &status);

        int idx = 0;
        int layertype = recvbuffer[idx];  idx++;
        int index     = recvbuffer[idx];  idx++;
        int dimIn     = recvbuffer[idx];  idx++;
        int dimOut    = recvbuffer[idx];  idx++;
        int dimBias   = recvbuffer[idx];  idx++;
        int activ     = recvbuffer[idx];  idx++;
        int nDesign   = recvbuffer[idx];  idx++;
        int gamma     = recvbuffer[idx];  idx++;
        switch (layertype)
        {
            case Layer::OPENZERO:
                recvlayer = new OpenExpandZero(dimIn, dimOut);
                break;
            case Layer::OPENDENSE:
                recvlayer = new OpenDenseLayer(dimIn, dimOut, activ, gamma);
                break;
            case Layer::DENSE:
                recvlayer = new DenseLayer(index, dimIn, dimOut, 1.0, activ, gamma);
                break;
            case Layer::CLASSIFICATION:
                recvlayer = new ClassificationLayer(index, dimIn, dimOut, gamma);
                break;
            default: 
                printf("\n\n ERROR while unpacking a buffer: Layertype unknown!!\n\n"); 
        }
        double *design   = new double[nDesign];
        double *gradient = new double[nDesign];
        recvlayer->initialize(design, gradient, 0.0);

        int nweights     = nDesign - dimBias;
        int nbias        = nDesign - dimIn * dimOut;
        for (int i = 0; i < nweights; i++)
        {
            recvlayer->getWeights()[i]    = recvbuffer[idx]; idx++;
            recvlayer->getWeightsBar()[i] = recvbuffer[idx]; idx++;
        }
        for (int i = 0; i < nbias; i++)
        {
            recvlayer->getBias()[i]    = recvbuffer[idx];   idx++;
            recvlayer->getBiasBar()[i] = recvbuffer[idx];   idx++;
        }

    }

    /* Free the buffer */
    delete [] sendbuffer;
    delete [] recvbuffer;

    return recvlayer;
}