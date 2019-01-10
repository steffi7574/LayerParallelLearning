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

Network::Network(int     StartLayerID, 
                 int     EndLayerID,
                 Config* config)
{
    /* Initilizize */
    startlayerID     = StartLayerID;
    endlayerID       = EndLayerID;
    nlayers_local    = endlayerID - startlayerID + 1;
    nlayers_global   = config->nlayers;
    openlayer        = NULL;
    layers           = NULL;
    layer_left       = NULL;
    layer_right      = NULL;

    nchannels        = config->nchannels;
    dt               = (config->T) / (MyReal)(config->nlayers-2);  // nlayers-2 = nhiddenlayers
    loss             = 0.0;
    accuracy         = 0.0;


    /* Sanity check */
    if (config->nfeatures > nchannels ||
        config->nclasses  > nchannels)
    {
        printf("ERROR! Choose a wider netword!\n");
        printf(" -- nFeatures = %d\n", config->nfeatures);
        printf(" -- nChannels = %d\n", config->nchannels);
        printf(" -- nClasses = %d\n", config->nclasses);
        exit(1);
    }


    /* --- Create the layers --- */
    ndesign_loc = 0;

    /* Create Opening layer */
    if (startlayerID == 0)
    {
        /* Create the opening layer */
        int index = -1;
        openlayer = createLayer(index, config);
        ndesign_loc += openlayer->getnDesign();
        // printf("Create opening layer %d, ndesign_loc %d \n", index, openlayer->getnDesign());
    }

   /* Create intermediate layers and classification layer */
    layers  = new Layer*[nlayers_local];
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        /* Create a layer at time step ilayer. Local storage at ilayer - startlayerID */
        int storeID = getLocalID(ilayer);
        layers[storeID] = createLayer(ilayer, config);
        ndesign_loc += layers[storeID]->getnDesign();
        // printf("creating hidden/class layer %d/%d, ndesign_loc%d\n", ilayer, nlayers_local, layers[storeID]->getnDesign());
    }


    /* Allocate memory for network design and gradient variables */
    design   = new MyReal[ndesign_loc];
    gradient = new MyReal[ndesign_loc];

    /* Create left neighbouring layer */
    int leftID = startlayerID - 1;
    layer_left = createLayer(leftID, config);

    /* Create right neighbrouing layer */
    int rightID = endlayerID + 1;
    layer_right = createLayer(rightID, config);
}             

  



Network::~Network()
{
    /* Delete openlayer */
    if (openlayer != NULL) delete openlayer;

    /* Delete intermediate and classification layers */
    for (int ilayer = 0; ilayer < nlayers_local; ilayer++)
    {
        delete layers[ilayer];
    }
    delete [] layers;

    /* Delete design and gradient */
    delete [] design;
    delete [] gradient;

    /* Delete neighbouring layer information */
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

int Network::getnLayersGlobal() { return nlayers_global; }

MyReal Network::getDT() { return dt; }

int Network::getLocalID(int ilayer) 
{
    int idx = ilayer - startlayerID;
    return idx;
}

MyReal Network::getLoss() { return loss; }

MyReal Network::getAccuracy() { return accuracy; }

int Network::getnDesignLocal() { return ndesign_loc; }

MyReal* Network::getDesign() { return design; }
       
MyReal* Network::getGradient() { return gradient; }

int Network::getStartLayerID() { return startlayerID; }
int Network::getEndLayerID()   { return endlayerID; }

Layer* Network::createLayer(int     index,
                            Config *config)
{
    Layer* layer = 0;
    if (index == -1)  // Opening layer
    {
        switch ( config->network_type )
{
            case DENSE: 
                if (config->weights_open_init == 0.0)
                {
                   layer  = new OpenExpandZero(config->nfeatures, nchannels);
                }
                else
                {
                   layer = new OpenDenseLayer(config->nfeatures, nchannels, config->activation, config->gamma_tik);
                }
                break;
            case CONVOLUTIONAL:
                /**< (Weight_open_init == 0.0) not needed for convolutional layers*/
                if (config->openlayer_type == 0)
                {
                   layer = new OpenConvLayer(config->nfeatures, nchannels);
                }
                else if (config->openlayer_type == 1)
                {
                   layer = new OpenConvLayerMNIST(config->nfeatures, nchannels);
                }
                break;
        }
    }
    else if (0 <= index && index < nlayers_global-2) // Intermediate layer
    {
        switch ( config->network_type )
        {
            case DENSE:
                layer = new DenseLayer(index, nchannels, nchannels, dt, config->activation, config->gamma_tik, config->gamma_ddt);
                break;
            case CONVOLUTIONAL:
                // TODO: Fix
                int convolution_size = 3;
                layer = new ConvLayer(index, nchannels, nchannels, convolution_size, nchannels/config->nfeatures, dt, config->activation, config->gamma_tik, config->gamma_ddt);
                break;
        }
    }
    else if (index == nlayers_global-2) // Classification layer 
    {
        layer = new ClassificationLayer(index, nchannels, config->nclasses, config->gamma_class);
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

    if (layerindex == -1)  // opening layer
    {
        layer = openlayer;
    }
    else if (layerindex == startlayerID - 1)  
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

int Network::getnDesignLayermax()
{
    int ndesignlayer;
    int max = 0;

    /* Loop over all local layers */
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        if (ilayer < nlayers_global-2) // excludes classification layer
        {
            /* Update maximum */
            ndesignlayer = layers[getLocalID(ilayer)]->getnDesign();
            if ( ndesignlayer > max)  max = ndesignlayer;
        }
    }

    return max;
}

void Network::initialize(Config *config)
{
    MyReal factor;
    char   filename[255];

    /* Initialize  the layer weights and bias  */
    int istart = 0;

    /* Opening layer on first processor */
    if (startlayerID == 0)
    {
        /* Set memory location for design and scale design by the factor */
        factor = config->weights_open_init;
        openlayer->initialize(&(design[istart]), &(gradient[istart]), factor);

        /* if set, overwrite opening design from file */
        if (strcmp(config->weightsopenfile, "NONE") != 0)
        {
           sprintf(filename, "%s/%s", config->datafolder, config->weightsopenfile);
           read_vector(filename, &(design[istart]), openlayer->getnDesign());
        }

        /* Increase counter */ 
        istart += openlayer->getnDesign();
    }

    /* Intermediate (hidden) and classification layers */
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        if (ilayer < nlayers_global-1) // Intermediate layer
        {
            factor = config->weights_init;
        }
        else // Classification layer 
        {
            factor = config->weights_class_init;
        }

        /* Set memory location and scale the current design by the factor */
        int storeID = getLocalID(ilayer);
        layers[storeID]->initialize(&(design[istart]), &(gradient[istart]), factor);

        /* if set, overwrite classification design from file */
        if (ilayer == nlayers_global-1)
        {
            if (strcmp(config->weightsclassificationfile, "NONE") != 0)
            {
                sprintf(filename, "%s/%s", config->datafolder, config->weightsclassificationfile);
                read_vector(filename, &(design[istart]), layers[storeID]->getnDesign());
            }
        }

        /* Increase the counter */
        istart += layers[storeID]->getnDesign();
    }

    /* Create and initialize left neighbouring layer, if exists */
    if (layer_left != NULL)
    {
        MyReal *left_design   = new MyReal[layer_left->getnDesign()];
        MyReal *left_gradient = new MyReal[layer_left->getnDesign()];
        layer_left->initialize(left_design, left_gradient, 0.0);
    }


    /* Create and initialize right neighbouring layer, if exists */
    if (layer_right != NULL)
    {
        MyReal *right_design   = new MyReal[layer_right->getnDesign()];
        MyReal *right_gradient = new MyReal[layer_right->getnDesign()];
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
    int size_left = -1; 
    int size_right = -1; 

    MyReal* sendlast  = 0;
    MyReal* recvlast  = 0; 
    MyReal* sendfirst = 0;
    MyReal* recvfirst = 0;

    /* --- All but the first process receive the last layer from left neighbour --- */
    if (myid > 0)
    {
        /* Receive from left neighbour */
        int source = myid - 1;

        size_left = layer_left->getnDesign();
        recvlast  = new MyReal[size_left];

        MPI_Irecv(recvlast, size_left, MPI_MyReal, source, 0, comm, &recvlastreq);
    }

    /* --- All but the last process sent their last layer to right neighbour --- */
    if (myid < comm_size-1)
    {
        size_left = layers[getLocalID(endlayerID)]->getnDesign();
        sendlast  = new MyReal[size_left];
        
        /* Pack the last layer into a buffer */
        layers[getLocalID(endlayerID)]->packDesign(sendlast, size_left);

       /* Send to right neighbour */
        int receiver = myid + 1;
        MPI_Isend(sendlast, size_left, MPI_MyReal, receiver, 0, comm, &sendlastreq);
    }

    /* --- All but the last processor recv the first layer from the right neighbour --- */
    if (myid < comm_size - 1)
    {
        /* Receive from right neighbour */
        int source = myid + 1;

        size_right = layer_right->getnDesign();
        recvfirst  = new MyReal[size_right];

        MPI_Irecv(recvfirst, size_right, MPI_MyReal, source, 1, comm, &recvfirstreq);
    }


    /* --- All but the first processor send their first layer to the left neighbour --- */
    if (myid > 0)
    {
        size_right = layers[getLocalID(startlayerID)]->getnDesign();
        sendfirst  = new MyReal[size_right];

        /* Pack the first layer into a buffer */
        layers[getLocalID(startlayerID)]->packDesign(sendfirst, size_right);

        /* Send to left neighbour */
        int receiver = myid - 1;
        MPI_Isend(sendfirst, size_right, MPI_MyReal, receiver, 1, comm, &sendfirstreq);
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
    if(sendlast!=0) delete [] sendlast;
    if(recvlast!=0) delete [] recvlast;
    if(sendfirst!=0) delete [] sendfirst;
    if(recvfirst!=0) delete [] recvfirst;
}


void Network::evalClassification(DataSet* data, 
                                 MyReal** state,
                                 MyReal*  loss_ptr, 
                                 MyReal*  accuracy_ptr,
                                 int      output)
{
    MyReal *tmpstate = new MyReal[nchannels];
    
    MyReal loss, accuracy;
    int    class_id;
    int    success, success_local;
    FILE*  classfile;
    ClassificationLayer* classificationlayer;
    

    /* Get classification layer */
    classificationlayer = dynamic_cast<ClassificationLayer*>(getLayer(nlayers_global - 2));
    if (classificationlayer == NULL) 
    {
        printf("\n ERROR: Network can't access classification layer!\n\n");
        exit(1);
    }

    /* open file for printing predicted file */
    if (output) classfile = fopen("classprediction.dat", "w");

    loss    = 0.0;
    success = 0;
    for (int iex = 0; iex < data->getnElements(); iex++)
    {
        /* Copy values so that they are not overwrittn (they are needed for adjoint)*/
        for (int ic = 0; ic < nchannels; ic++)
        {
            tmpstate[ic] = state[iex][ic];
        }
        /* Apply classification on tmpstate */
        classificationlayer->applyFWD(tmpstate);
        /* Evaluate Loss */
        loss          += classificationlayer->crossEntropy(tmpstate, data->getLabel(iex));
        success_local  = classificationlayer->prediction(tmpstate, data->getLabel(iex), &class_id);
        success       += success_local;
        if (output) fprintf(classfile, "%d   %d\n", class_id, success_local );
    }
    loss     = 1. / data->getnElements() * loss;
    accuracy = 100.0 * ( (MyReal) success ) / data->getnElements();
    // printf("Classification %d: %1.14e using layer %1.14e state %1.14e tmpstate[0] %1.14e\n", getIndex(), loss, weights[0], state[1][1], tmpstate[0]);

    /* Return */
    *loss_ptr      = loss;
    *accuracy_ptr  = accuracy;

    if (output) fclose(classfile);
    if (output) printf("Prediction file written: classprediction.dat\n");

    delete [] tmpstate;
}       


void Network::evalClassification_diff(DataSet* data, 
                                      MyReal** primalstate,
                                      MyReal** adjointstate,
                                      int      compute_gradient)
{
    MyReal *tmpstate = new MyReal[nchannels];
    ClassificationLayer* classificationlayer;
    
    /* Get classification layer */
    classificationlayer = dynamic_cast<ClassificationLayer*>(getLayer(nlayers_global - 2));
    if (classificationlayer == NULL) 
    {
        printf("\n ERROR: Network can't access classification layer!\n\n");
        exit(1);
    }

    int    nexamples = data->getnElements();
    MyReal loss_bar = 1./nexamples; 
    
    for (int iex = 0; iex < nexamples; iex++)
    {
        /* Recompute the Classification */
        for (int ic = 0; ic < nchannels; ic++)
        {
            tmpstate[ic] = primalstate[iex][ic];
        }
        classificationlayer->applyFWD(tmpstate);

        /* Derivative of Loss and classification. */
        classificationlayer->crossEntropy_diff(tmpstate, adjointstate[iex], data->getLabel(iex), loss_bar);
        classificationlayer->applyBWD(primalstate[iex], adjointstate[iex], compute_gradient);
    }
    // printf("Classification_diff %d using layer %1.14e state %1.14e tmpstate %1.14e biasbar[dimOut-1] %1.14e\n", getIndex(), weights[0], primalstate[1][1], tmpstate[0], bias_bar[dim_Out-1]);

    delete [] tmpstate;

}

void Network::updateDesign(MyReal   stepsize,
                           MyReal  *direction,
                           MPI_Comm comm)
{
    /* Update design locally on this network-block */
    for (int id = 0; id < ndesign_loc; id++)
    {
        design[id] += stepsize * direction[id];
    }

    /* Communicate design across neighbouring processors (ghostlayers) */
    MPI_CommunicateNeighbours(comm);
}                   