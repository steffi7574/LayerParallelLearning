#include "braid_wrapper.hpp"
// #include "codi.hpp"
// #include "lib.hpp"


void braid_SetConfigOptions(braid_Core core, 
                            Config*    config)
{
    braid_SetMaxLevels(core, config->braid_maxlevels);
    braid_SetMinCoarse(core, config->braid_mincoarse);
    braid_SetPrintLevel( core, config->braid_printlevel);
    braid_SetCFactor(core,  0, config->braid_cfactor0);
    braid_SetCFactor(core, -1, config->braid_cfactor);
    braid_SetAccessLevel(core, config->braid_accesslevel);
    braid_SetMaxIter(core, config->braid_maxiter);
    braid_SetSkip(core, config->braid_setskip);
    if (config->braid_fmg){
        braid_SetFMG(core);
    }
    braid_SetNRelax(core, -1, config->braid_nrelax);
    braid_SetNRelax(core,  0, config->braid_nrelax0);
    braid_SetAbsTol(core, config->braid_abstol);
}

int GetTimeStepIndex(braid_App app, 
                     MyReal    t)
{

    /* Round to the closes integer */
    int ts = round(t / app->network->getDT()) ;
    return ts;
}

int GetPrimalIndex(braid_App app,
                   int       ts)
{
    int idx = app->network->getnLayersGlobal() - 2 - ts;
    return idx;
}       

int 
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
    int    ts_start, ts_stop;
    MyReal tstart, tstop;
    MyReal deltaT;

    int nexamples = app->data->getnElements();
   
    /* Get the time-step size and current time index*/
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    ts_start = GetTimeStepIndex(app, tstart); 
    ts_stop  = GetTimeStepIndex(app, tstop); 
    deltaT   = tstop - tstart;

    /* Set time step size */
    u->layer->setDt(deltaT);

    // printf("%d: step %d,%f -> %d, %f layer %d using %1.14e state %1.14e, %d\n", app->myid, ts_start, tstart, ts_stop, tstop, u->layer->getIndex(), u->layer->getWeights()[3], u->state[1][1], u->layer->getnDesign());

    /* apply the layer for all examples */
    for (int iex = 0; iex < nexamples; iex++)
    {
        /* Apply the layer */
        u->layer->applyFWD(u->state[iex]);
    }


    /* Free the layer, if it has just been send to this processor */
    if (u->sendflag > 0.0)
    {
        delete [] u->layer->getWeights();
        delete [] u->layer->getWeightsBar();
        delete u->layer;
    }
    u->sendflag = -1.0;

    /* Move the layer pointer of u forward to that of tstop */
    u->layer = app->network->getLayer(ts_stop);


    // /* no refinement */
    braid_StepStatusSetRFactor(status, 1);
 
    return 0;
}   


int
my_Init(braid_App     app,
        MyReal        t,
        braid_Vector *u_ptr)
{
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();


    /* Initialize the state */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new MyReal*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new MyReal[nchannels];
        for (int ic = 0; ic < nchannels; ic++)
        {
            u->state[iex][ic] = 0.0;
        }
    }

    /* Apply the opening layer */ 
    if (t == 0)
    {
        Layer* openlayer = app->network->getLayer(-1);
        // printf("%d: Init %f: layer %d using %1.14e state %1.14e, %d\n", app->myid, t, openlayer->getIndex(), openlayer->getWeights()[3], u->state[1][1], openlayer->getnDesign());
        for (int iex = 0; iex < nexamples; iex++)
        {
            /* set example */
            openlayer->setExample(app->data->getExample(iex));

            /* Apply the layer */
            openlayer->applyFWD(u->state[iex]);
        }
    }

    /* Set the layer pointer */
    if (t >=0 ) // this should always be the case...
    {
        /* Store a pointer to the layer design */
        int ilayer  = GetTimeStepIndex(app, t);
        u->layer = app->network->getLayer(ilayer);  // this is either a hidden or a classification layer
    }
    u->sendflag = -1.0;
    
    *u_ptr = u;
    return 0;
}


int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
    // my_Vector *v;
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();
 
    /* Allocate the vector */
    my_Vector* v = (my_Vector *) malloc(sizeof(my_Vector));
    v->state = new MyReal*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        v->state[iex] = new MyReal[nchannels];
        for (int ic = 0; ic < nchannels; ic++)
        {
            v->state[iex][ic] = u->state[iex][ic];
        }
    }
    v->layer      = u->layer;
    v->sendflag = u->sendflag;

    *v_ptr = v;

    return 0;
}


int
my_Free(braid_App    app,
        braid_Vector u)
{
    int nexamples = app->data->getnElements();

    for (int iex = 0; iex < nexamples; iex++)
    {
        delete [] u->state[iex];
    }
    delete [] u->state;
    free(u);

   return 0;
}


int
my_Sum(braid_App     app,
       MyReal        alpha,
       braid_Vector  x,
       MyReal        beta,
       braid_Vector  y)
{
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();

    for (int iex = 0; iex < nexamples; iex++)
    {
        for (int ic = 0; ic < nchannels; ic++)
        {
           (y->state)[iex][ic] = alpha*(x->state)[iex][ic] + beta*(y->state)[iex][ic];
        }
    }

   return 0;
}

int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               MyReal       *norm_ptr)
{
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();

    MyReal dot = 0.0;
    for (int iex = 0; iex < nexamples; iex++)
    {
        dot += vecdot(nchannels, u->state[iex], u->state[iex]);
    }
   *norm_ptr = sqrt(dot) / nexamples;

   return 0;
}



int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus)
{
    printf("my_Access: To be implemented...\n");

    return 0;
}


int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus)
{
    int nchannels        = app->network->getnChannels();
    int nexamples        = app->data->getnElements();
    int ndesign_layermax = app->ndesign_layermax;
   
    /* Gather number of variables */
    int nuvector     = nchannels*nexamples;
    int nlayerinfo   = 12;
    int nlayerdesign = ndesign_layermax;

    /* Set the size */
    *size_ptr = (nuvector + nlayerinfo + nlayerdesign) * sizeof(MyReal);

    return 0;
}


int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{
    int size;
    MyReal *dbuffer   = (MyReal*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();
    
    /* Store network state */
    int idx = 0;
    for (int iex = 0; iex < nexamples; iex++)
    {
        for (int ic = 0; ic < nchannels; ic++)
        {
           dbuffer[idx] = (u->state)[iex][ic];
           idx++;
        }
    }
    size = nchannels*nexamples*sizeof(MyReal);

    int nweights = u->layer->getnWeights();
    int nbias    = u->layer->getDimBias();

    dbuffer[idx] = u->layer->getType();       idx++;
    dbuffer[idx] = u->layer->getIndex();      idx++;
    dbuffer[idx] = u->layer->getDimIn();      idx++;
    dbuffer[idx] = u->layer->getDimOut();     idx++;
    dbuffer[idx] = u->layer->getDimBias();    idx++;
    dbuffer[idx] = u->layer->getnWeights();   idx++;
    dbuffer[idx] = u->layer->getActivation(); idx++;
    dbuffer[idx] = u->layer->getnDesign();    idx++;
    dbuffer[idx] = u->layer->getGammaTik();   idx++;
    dbuffer[idx] = u->layer->getGammaDDT();   idx++;
    dbuffer[idx] = u->layer->getnConv();      idx++;
    dbuffer[idx] = u->layer->getCSize();      idx++;
    for (int i = 0; i < nweights; i++)
    {
        dbuffer[idx] = u->layer->getWeights()[i];     idx++;
        // dbuffer[idx] = u->layer->getWeightsBar()[i];  idx++;
    }
    for (int i = 0; i < nbias; i++)
    {
        dbuffer[idx] = u->layer->getBias()[i];     idx++;
        // dbuffer[idx] = u->layer->getBiasBar()[i];  idx++;
    }
    size += (12 + (nweights+nbias))*sizeof(MyReal);

    braid_BufferStatusSetSize( bstatus, size);
 
   return 0;
}



int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus)
{

    MyReal *dbuffer   = (MyReal*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();
    Layer *tmplayer = 0;
    
    //  /* Allocate the vector */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new MyReal*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new MyReal[nchannels];
    }

    /* Unpack the buffer */
    int idx = 0;
    for (int iex = 0; iex < nexamples; iex++)
    {
        for (int ic = 0; ic < nchannels; ic++)
        {
           u->state[iex][ic] = dbuffer[idx]; 
           idx++;
        }
    }

    /* Receive and initialize a layer. Set the sendflag */
    int layertype = dbuffer[idx];  idx++;
    int index     = dbuffer[idx];  idx++;
    int dimIn     = dbuffer[idx];  idx++;
    int dimOut    = dbuffer[idx];  idx++;
    int dimBias   = dbuffer[idx];  idx++;
    int nweights  = dbuffer[idx];  idx++;
    int activ     = dbuffer[idx];  idx++;
    int nDesign   = dbuffer[idx];  idx++;
    int gammatik  = dbuffer[idx];  idx++;
    int gammaddt  = dbuffer[idx];  idx++;
    int nconv     = dbuffer[idx];  idx++;
    int csize     = dbuffer[idx];  idx++;

    /* layertype decides on which layer should be created */
    switch (layertype)
    {
        case Layer::OPENZERO:
            tmplayer = new OpenExpandZero(dimIn, dimOut);
            break;
        case Layer::OPENDENSE:
            tmplayer = new OpenDenseLayer(dimIn, dimOut, activ, gammatik);
            break;
        case Layer::DENSE:
            tmplayer = new DenseLayer(index, dimIn, dimOut, 1.0, activ, gammatik, gammaddt);
            break;
        case Layer::CLASSIFICATION:
            tmplayer = new ClassificationLayer(index, dimIn, dimOut, gammatik);
            break;
        case Layer::OPENCONV:
            tmplayer = new OpenConvLayer(dimIn, dimOut);
            break;
        case Layer::OPENCONVMNIST:
            tmplayer = new OpenConvLayerMNIST(dimIn, dimOut);
            break;
        case Layer::CONVOLUTION:
            tmplayer = new ConvLayer(index, dimIn, dimOut, csize, nconv, 1.0, activ, gammatik, gammaddt);
            break;
        default: 
            printf("\n\n ERROR while unpacking a buffer: Layertype unknown!!\n\n"); 
    }

    /* Allocate design and gradient */
    MyReal *design   = new MyReal[nDesign];
    MyReal *gradient = new MyReal[nDesign];
    tmplayer->initialize(design, gradient, 0.0);
    /* Set the weights */
    for (int i = 0; i < nweights; i++)
    {
        tmplayer->getWeights()[i]    = dbuffer[idx]; idx++;
    }
    for (int i = 0; i < dimBias; i++)
    {
        tmplayer->getBias()[i]    = dbuffer[idx];   idx++;
    }
    u->layer = tmplayer;
    u->sendflag = 1.0;

    *u_ptr = u;
    return 0;
}




int 
my_Step_Adj(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     fstop,
            braid_Vector     u,
            braid_StepStatus status)
{
    int    ts_start, ts_stop;
    int    level, compute_gradient;
    MyReal tstart, tstop;
    MyReal deltaT;
    int    finegrid  = 0;
    int    primaltimestep;
    braid_BaseVector ubaseprimal;
    braid_Vector     uprimal;
    int    nexamples = app->data->getnElements();

    /* Update gradient only on the finest grid */
    braid_StepStatusGetLevel(status, &level);
    if (level == 0) compute_gradient = 1;
    else            compute_gradient = 0;
   
    /* Get the time-step size and current time index*/
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    ts_start       = GetTimeStepIndex(app, tstart); 
    ts_stop        = GetTimeStepIndex(app, tstop); 
    deltaT         = tstop - tstart;
    primaltimestep = GetPrimalIndex(app, ts_stop); 


    /* Get the primal vector from the primal core */
    _braid_UGetVectorRef(app->primalcore, finegrid, primaltimestep, &ubaseprimal);
    uprimal = ubaseprimal->userVector;

    /* Reset gradient before the update */
    if (compute_gradient) uprimal->layer->resetBar();

    /* Take one step backwards, updates adjoint state and gradient, if desired. */
    uprimal->layer->setDt(deltaT);
    for (int iex = 0; iex < nexamples; iex++)
    {
        uprimal->layer->applyBWD(uprimal->state[iex], u->state[iex], compute_gradient); 
    }

    // printf("%d: level %d step_adj %d->%d using layer %d,%1.14e, primal %1.14e, adj %1.14e, grad[0] %1.14e, %d\n", app->myid, level, ts_start, ts_stop, uprimal->layer->getIndex(), uprimal->layer->getWeights()[3], uprimal->state[1][1], u->state[1][1], uprimal->layer->getWeightsBar()[0],  uprimal->layer->getnDesign());

    /* Derivative of DDT-Regularization */
    if (compute_gradient) 
    {
        Layer* prev = app->network->getLayer(primaltimestep - 1); 
        Layer* next = app->network->getLayer(primaltimestep + 1); 
        uprimal->layer->evalRegulDDT_diff(prev, next, app->network->getDT());
    }        

    /* Derivative of tikhonov */
    if (compute_gradient) uprimal->layer->evalTikh_diff(1.0);


    /* no refinement */
    braid_StepStatusSetRFactor(status, 1);

    return 0;
}

int
my_Init_Adj(braid_App     app,
            MyReal        t,
            braid_Vector *u_ptr)
{
    braid_BaseVector uprimal;
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();
    MyReal *aux     = new MyReal[nchannels];

    int finegrid         = 0;
    int ilayer           = GetTimeStepIndex(app, t);
    int primaltimestep   = GetPrimalIndex(app, ilayer);


    // printf("%d: Init %d (primaltimestep %d)\n", app->myid, ilayer, primaltimestep);

    /* Allocate the adjoint vector and set to zero */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new MyReal*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new MyReal[nchannels];
        for (int ic = 0; ic < nchannels; ic++)
        {
            u->state[iex][ic] = 0.0;
        }
    }
    u->layer    = NULL;
    u->sendflag = -1.0;

    /* Adjoint initial (i.e. terminal) condition is derivative of classification layer */
    if (t==0)
    {
        /* Get the primal vector */
        _braid_UGetVectorRef(app->primalcore, finegrid, primaltimestep, &uprimal);
        MyReal** primalstate = uprimal->userVector->state;
        Layer* layer = uprimal->userVector->layer;

        /* Reset the gradient before updating it */
        layer->resetBar();

        /* Derivative of classification */
        layer->evalClassification_diff(app->data, primalstate, u->state, 1);


        /* Derivative of tikhonov regularization) */
        layer->evalTikh_diff(1.0);
 
    //    printf("%d: Init_adj Loss at %d, using %1.14e, primal %1.14e, adj %1.14e, grad[0] %1.14e\n", app->myid, layer->getIndex(), layer->getWeights()[0], primalstate[1][1], u->state[1][1], layer->getWeightsBar()[0]);
    }

    delete [] aux;

    *u_ptr = u;
    return 0;
}

int
my_BufSize_Adj(braid_App           app,
               int                 *size_ptr,
               braid_BufferStatus  bstatus)
{
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();
   
    *size_ptr = nchannels*nexamples*sizeof(MyReal);
    return 0;
}



int
my_BufPack_Adj(braid_App           app,
               braid_Vector        u,
               void               *buffer,
               braid_BufferStatus  bstatus)
{
    int size;
    MyReal *dbuffer   = (MyReal*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();
    
    /* Store network state */
    int idx = 0;
    for (int iex = 0; iex < nexamples; iex++)
    {
        for (int ic = 0; ic < nchannels; ic++)
        {
           dbuffer[idx] = (u->state)[iex][ic];
           idx++;
        }
    }
    size = nchannels*nexamples*sizeof(MyReal);

    braid_BufferStatusSetSize( bstatus, size);
 
   return 0;
}



int
my_BufUnpack_Adj(braid_App           app,
                 void               *buffer,
                 braid_Vector       *u_ptr,
                 braid_BufferStatus  bstatus)
{

    MyReal *dbuffer   = (MyReal*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->data->getnElements();
    
    //  /* Allocate the vector */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new MyReal*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new MyReal[nchannels];
    }

    /* Unpack the buffer */
    int idx = 0;
    for (int iex = 0; iex < nexamples; iex++)
    {
        for (int ic = 0; ic < nchannels; ic++)
        {
           u->state[iex][ic] = dbuffer[idx]; 
           idx++;
        }
    }
    u->layer    = NULL;
    u->sendflag = -1.0;

 
    *u_ptr = u;
    return 0;
}

void
evalInit(braid_Core core,
         braid_App   app)
{
    Layer* openlayer = app->network->getLayer(-1);
    int    nexamples = app->data->getnElements();
    braid_BaseVector ubase;
    braid_Vector     u;


    /* Apply initial condition if warm_restart (otherwise it is set in my_Init() */
    /* can not be set here if !(warm_restart) because braid_grid is created only in braid_drive(). */
    if (_braid_CoreElt(core, warm_restart)) 
    {
        /* Get vector at t == 0 */
        _braid_UGetVectorRef(core, 0, 0, &ubase);
        if (ubase != NULL) // only true on first processor 
        {
            u = ubase->userVector;

            /* Apply opening layer */
            for (int iex = 0; iex < nexamples; iex++)
            {
                /* set example */
                openlayer->setExample(app->data->getExample(iex));

                /* Apply the layer */
                openlayer->applyFWD(u->state[iex]);
            } 
        }
    }
}


void
evalObjective(braid_Core core,
              braid_App   app,
              MyReal     *objective_ptr,
              MyReal     *loss_ptr,
              MyReal     *accuracy_ptr)
{
    MyReal objective = 0.0;
    MyReal regul     = 0.0;
    MyReal loss      = 0.0;
    MyReal accuracy  = 0.0;
    MyReal loss_loc  = 0.0; 
    MyReal accur_loc = 0.0; 
    braid_BaseVector ubase;
    braid_Vector     u;
    Layer* layer;

    /* Get range of locally stored layers */
    int startlayerID = app->network->getStartLayerID();
    int endlayerID   = app->network->getEndLayerID();
    if (startlayerID == 0) startlayerID -= 1; // this include opening layer (id = -1) at first processor 

    /* Iterate over the local layers */
    for (int ilayer = startlayerID; ilayer <= endlayerID; ilayer++)
    {
        /* Get the layer */
        layer = app->network->getLayer(ilayer); 

         /* Tikhonov - Regularization*/
        regul += layer->evalTikh();

        /* DDT - Regularization on intermediate layers */
        regul += layer->evalRegulDDT(app->network->getLayer(ilayer-1), app->network->getDT());

        /* At last layer: Classification and Loss evaluation */ 
        if (ilayer == app->network->getnLayersGlobal()-2)
        {
            _braid_UGetVectorRef(core, 0, ilayer, &ubase);
            u = ubase->userVector;
            layer->evalClassification(app->data, u->state, &loss_loc, &accur_loc,0);
            loss     += loss_loc;
            accuracy += accur_loc;
        }
        // printf("%d: layerid %d using %1.14e, tik %1.14e, ddt %1.14e, loss %1.14e\n", app->myid, layer->getIndex(), layer->getWeights()[0], regultik, regulddt, loss_loc);
    }


    /* Collect objective function from all processors */
    MyReal myobjective = loss + regul;
    MPI_Allreduce(&myobjective, &objective, 1, MPI_MyReal, MPI_SUM, MPI_COMM_WORLD);

    /* Return */
    *objective_ptr = objective;
    *accuracy_ptr  = accuracy;
    *loss_ptr      = loss;

}          


void
evalObjectiveDiff(braid_Core core_adj,
                  braid_App  app)
{

    braid_BaseVector  ubaseprimal, ubaseadjoint;
    braid_Vector      uprimal, uadjoint;
    int warm_restart = _braid_CoreElt(core_adj, warm_restart);

    /* Only gradient for primal time step N here. Other time steps are in my_Step_adj. */

    /* If warm_restart: set adjoint initial condition here. Otherwise it's set in my_Init_Adj */
    /* It can not be done here if drive() has not been called before, because the braid grid is allocated only at the beginning of drive() */
    if (warm_restart)
    {
        /* Get primal and adjoint state */
        _braid_UGetVectorRef(app->primalcore, 0, GetPrimalIndex(app, 0), &ubaseprimal);
        _braid_UGetVectorRef(core_adj, 0, 0, &ubaseadjoint);

        if (ubaseprimal != NULL && ubaseadjoint !=NULL)  // this is the case at first primal and last adjoint time step  
        {
            uprimal  = ubaseprimal->userVector;
            uadjoint = ubaseadjoint->userVector;

            /* Reset the gradient before updating it */
            uprimal->layer->resetBar();

            // printf("%d: objective_diff at ilayer %d using %1.14e primal %1.14e\n", app->myid, uprimal->layer->getIndex(), uprimal->layer->getWeights()[0], uprimal->state[1][1]);

            /* Derivative of classification */
            uprimal->layer->evalClassification_diff(app->data, uprimal->state, uadjoint->state, 1);

            /* Derivative of tikhonov regularization) */
            uprimal->layer->evalTikh_diff(1.0);
        }
    }
}              


void 
evalInitDiff(braid_Core core_adj,
             braid_App  app)
{
    Layer* openlayer = app->network->getLayer(-1);
    int    nexamples = app->data->getnElements();
    braid_BaseVector ubase;

    /* Get \bar y^0 (which is the LAST xbraid vector, stored on proc 0) */
    _braid_UGetLast(core_adj, &ubase); 
    if (ubase != NULL) // This is true only on first processor (reverted ranks!)
    {
        openlayer->resetBar();
        
        /* Apply opening layer backwards for all examples */ 
        for (int iex = 0; iex < nexamples; iex++)
        {
            openlayer->setExample(app->data->getExample(iex));
            /* TODO: Don't feed applyBWD with NULL! */
            openlayer->applyBWD(NULL, ubase->userVector->state[iex], 1); 
        }

        // printf("%d: Init_diff layerid %d using %1.14e, adj %1.14e grad[0] %1.14e\n", app->myid, openlayer->getIndex(), openlayer->getWeights()[3], ubase->userVector->state[1][1], openlayer->getWeightsBar()[0] );

        /* Derivative of Tikhonov Regularization */
        openlayer->evalTikh_diff(1.0);
    }

}         
