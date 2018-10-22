#include "braid_wrapper.hpp"
// #include "codi.hpp"
// #include "lib.hpp"

int GetTimeStepIndex(braid_App app, 
                     double    t)
{

    /* Round to the closes integer */
    int ts = round(t / app->network->getDT()) ;
    return ts;
}

int GetPrimalIndex(braid_App app,
                   int       ts)
{
    int idx = app->network->getnLayers()-1 - ts;
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
    double tstart, tstop;
    double deltaT;

    int nexamples = app->nexamples;
   
    /* Get the time-step size and current time index*/
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    // braid_StepStatusGetTIndex(status, &ts);
    ts_start = GetTimeStepIndex(app, tstart); 
    ts_stop  = GetTimeStepIndex(app, tstop); 
    deltaT   = tstop - tstart;

    /* Set time step size */
    u->layer->setDt(deltaT);

    /* Get local storage ID */
    int storeID = app->network->getLocalID(ts_stop);
    printf("%d: step %d,%f -> %d, %f layer %d using %1.14e state %1.14e, %d\n", app->myid, ts_start, tstart, ts_stop, tstop, u->layer->getIndex(), u->layer->getWeights()[3], u->state[1][1], u->layer->getnDesign());

    /* apply the layer for all examples */
    for (int iex = 0; iex < nexamples; iex++)
    {
        /* On fist layer, set example */
        if (app->examples !=NULL) u->layer->setExample(app->examples[iex]);

        /* Apply the layer */
        u->layer->applyFWD(u->state[iex]);
    }


    /* Free the layer, if it has just been send to this processor */
    if (u->sendflag > 0.0)
    {
        delete [] u->layer->getWeights();
        delete [] u->layer->getWeightsBar();
    }
    u->sendflag = -1.0;

    /* Move the layer pointer of u forward to that of tstop */
    u->layer = app->network->layers[storeID];


    // /* no refinement */
    braid_StepStatusSetRFactor(status, 1);
 
    return 0;
}   


int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;


    /* Initialize the state */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new double*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new double[nchannels];
        for (int ic = 0; ic < nchannels; ic++)
        {
            u->state[iex][ic] = 0.0;
        }
    }


    /* Initialize the design (if adjoint: nonphysical time t=-1.0) */
    if (t >=0 )
    {
        int ilayer  = GetTimeStepIndex(app, t);
        int storeID = app->network->getLocalID(ilayer);

        /* Store a pointer to the layer design */
        u->layer = app->network->layers[storeID];
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
    int nexamples = app->nexamples;
 
    /* Allocate the vector */
    my_Vector* v = (my_Vector *) malloc(sizeof(my_Vector));
    v->state = new double*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        v->state[iex] = new double[nchannels];
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
    int nexamples = app->nexamples;

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
       double        alpha,
       braid_Vector  x,
       double        beta,
       braid_Vector  y)
{
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;

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
               double       *norm_ptr)
{
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;

    double dot = 0.0;
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
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
   
    *size_ptr = nchannels*nexamples*sizeof(double) + (8 + 2*(nchannels*nchannels+nchannels))*sizeof(double);
    return 0;
}



int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{
    int size;
    double layertype;
    double *dbuffer   = (double*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
    
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
    size = nchannels*nexamples*sizeof(double);

    /* Store Layer information for reconstructing the layer at different processor */
    if (u->layer->getIndex() == 0)
    {
        /* Encoding an opening layer */
        layertype = -1.0;
    }
    else 
    {
        /* Encoding a dense intermediate layer */
        layertype = 1.0;
    }
    int nweights = u->layer->getnDesign() - u->layer->getDimBias();
    int nbias    = u->layer->getnDesign() - u->layer->getDimIn() * u->layer->getDimOut();

    dbuffer[idx] = layertype;                 idx++;
    dbuffer[idx] = u->layer->getIndex();      idx++;
    dbuffer[idx] = u->layer->getDimIn();      idx++;
    dbuffer[idx] = u->layer->getDimOut();     idx++;
    dbuffer[idx] = u->layer->getDimBias();    idx++;
    dbuffer[idx] = u->layer->getActivation(); idx++;
    dbuffer[idx] = u->layer->getnDesign();    idx++;
    dbuffer[idx] = u->layer->getGamma();      idx++;
    for (int i = 0; i < nweights; i++)
    {
        dbuffer[idx] = u->layer->getWeights()[i];     idx++;
        dbuffer[idx] = u->layer->getWeightsBar()[i];  idx++;
        // printf("%d: send weights %d %1.14e %1.14e\n", app->myid, idx, dbuffer[idx-2], dbuffer[idx-1]);
    }
    for (int i = 0; i < nbias; i++)
    {
        dbuffer[idx] = u->layer->getBias()[i];     idx++;
        dbuffer[idx] = u->layer->getBiasBar()[i];  idx++;
    }
    size += (8 + 2*(nweights+nbias))*sizeof(double);

    // printf("%d: send weight %1.14e\n", app->myid, u->layer->getWeights()[3]);
    braid_BufferStatusSetSize( bstatus, size);
 
   return 0;
}



int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus)
{

    double *dbuffer   = (double*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
    Layer *tmplayer;
    
    //  /* Allocate the vector */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new double*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new double[nchannels];
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
    int activ     = dbuffer[idx];  idx++;
    int nDesign   = dbuffer[idx];  idx++;
    int gamma     = dbuffer[idx];  idx++;
    int nweights = nDesign - dimBias;
    int nbias    = nDesign - dimIn * dimOut;

    /* TODO: WARNING: THIS ONLY WORKS FOR DENSE LAYER TYPES!!! CHANGE FOR GENERAL LAYER SWITCH */
    if (layertype < 0)
    {
        /* Create an opening layer */
        tmplayer = new OpenDenseLayer(dimIn, dimOut, activ, gamma);
    }
    else
    {
        /* Create a dense layer */
        tmplayer = new DenseLayer(index, dimIn, dimOut, 1.0, activ, gamma);
    }
    /* Allocate design and gradient */
    double *design   = new double[nDesign];
    double *gradient = new double[nDesign];
    tmplayer->initialize(design, gradient, 0.0);
    /* Set the weights */
    for (int i = 0; i < nweights; i++)
    {
        tmplayer->getWeights()[i]    = dbuffer[idx]; idx++;
        tmplayer->getWeightsBar()[i] = dbuffer[idx]; idx++;
        // printf("%d: receive weights %d %1.14e %1.14e\n", app->myid, idx, dbuffer[idx-2], dbuffer[idx-1]);
    }
    for (int i = 0; i < nbias; i++)
    {
        tmplayer->getBias()[i]    = dbuffer[idx];   idx++;
        tmplayer->getBiasBar()[i] = dbuffer[idx];   idx++;
    }
    u->layer = tmplayer;
    u->sendflag = 1.0;

 
    *u_ptr = u;
    return 0;
}


// int 
// my_ObjectiveT(braid_App              app,
//               braid_Vector           u,
//               braid_ObjectiveStatus  ostatus,
//               double                *objective_ptr)
// {
//     int    success = 0;
//     double regul_tik = 0.0;
//     double regul_ddt = 0.0;
//     double loss      = 0.0;
//     double accuracy  = 0.0;

//     int nlayers   = app->network->getnLayers();
//     int nexamples = app->nexamples;

//     /* Get the time index*/
//     int ts;
//     braid_ObjectiveStatusGetTIndex(ostatus, &ts);
//     int ilayer = ts - 1;
//     if (ilayer < 0) 
//     {
//        *objective_ptr = 0.0;
//         return 0;
//     }

//     /* Tikhonov regularization */
//     regul_tik = app->network->layers[ilayer]->evalTikh();

//     /* ddt-regularization term */
//     if (ilayer > 1 && ilayer < nlayers - 1) 
//     {
//         regul_ddt = app->network->evalRegulDDT(app->network->layers[ilayer-1], app->network->layers[ilayer]);
//     }

 
//     /* Evaluate loss and accuracy */
//     for (int iex = 0; iex < nexamples; iex++)
//     {
//         loss += app->network->layers[ilayer]->evalLoss(u->state[iex], app->labels[iex]);

//         success += app->network->layers[ilayer]->prediction(u->state[iex], app->labels[iex]);
//     }
//     loss     = 1. / nexamples * loss;
//     accuracy = 100.0 * (double) success / nexamples;

//     /* Report to app */
//     if (ilayer == nlayers - 1)
//     {
//         app->loss     = loss;
//         app->accuracy = accuracy;
//     }

//     /* Compute objective function */
//     *objective_ptr = loss + regul_tik + regul_ddt;

    
//     return 0;
// }


double
evalObjectiveT(braid_App   app,
              braid_Vector u, 
              int          ilayer,
              double       *loss_ptr,
              double       *accuracy_ptr)
{
    int    success   = 0;
    double loss      = 0.0;
    double accuracy  = 0.0;
    double regul_tik = 0.0;
    double objective;
    int    nchannels = app->network->getnChannels();
    double *aux = new double[nchannels];

     /* Tikhonov */
    regul_tik = u->layer->evalTikh();

    /* TODO: DDT-REGULARIZATION */

    /* At last layer: Classification */ 
    if (ilayer == app->network->getnLayers()-1)
    {
        /* Sanity check */
        if (app->labels == NULL) printf("\n\n%d: ERROR! This should not happen! %d\n\n", app->myid, ilayer);

        for (int iex = 0; iex < app->nexamples; iex++)
        {
            /* Copy values so that they are not overwrittn (they are needed for adjoint)*/
            for (int ic = 0; ic < nchannels; ic++)
            {
                aux[ic] = u->state[iex][ic];
            }
            /* Apply classification on aux */
            u->layer->applyFWD(aux);
            /* Evaluate Loss */
            loss     += u->layer->evalLoss(aux, app->labels[iex]);
            success  += u->layer->prediction(aux, app->labels[iex]);
        }
        loss     = 1. / app->nexamples * loss;
        accuracy = 100.0 * (double) success / app->nexamples;
        printf("%d: Eval loss %d,%1.14e using %1.14e\n", app->myid, ilayer, loss, u->state[1][1]);
    }
  
    /* Return */
    objective     = loss + regul_tik;
    *accuracy_ptr = accuracy;
    *loss_ptr     = loss;

    delete [] aux;

    return objective;
}          


// int
// my_ObjectiveT_diff(braid_App            app,
//                   braid_Vector          u,
//                   braid_Vector          u_bar,
//                   braid_Real            f_bar,
//                   braid_ObjectiveStatus ostatus)
// {
//     double loss_bar, regul_tik_bar, regul_ddt_bar;
//     int nlayers   = app->network->getnLayers();
//     int nexamples = app->nexamples;

//     /* Get the time index*/
//     int ts;
//     braid_ObjectiveStatusGetTIndex(ostatus, &ts);
//     int ilayer = ts - 1;
//     if (ilayer < 0) 
//     {
//         return 0;
//     }

//     /* Derivative of objective function */
//     loss_bar      = f_bar;
//     regul_tik_bar = f_bar;
//     regul_ddt_bar = f_bar;

//     /* Derivative of loss function evaluation */
//     loss_bar = 1./nexamples * loss_bar;
//     for (int iex = 0; iex < nexamples; iex++)
//     {
//         app->network->layers[ilayer]->evalLoss_diff(u->state[iex], u_bar->state[iex], app->labels[iex], loss_bar);
//     }

//     /* Derivative of ddt-regularization term */
//     if (ilayer > 1 && ilayer < nlayers - 1) 
//     {
//         app->network->evalRegulDDT_diff(app->network->layers[ilayer-1], app->network->layers[ilayer], regul_ddt_bar);
//     }

//     /* Derivative of the tikhonov regularization term */
//     app->network->layers[ilayer]->evalTikh_diff(regul_tik_bar);

//     return 0;
// }

// int
// my_Step_diff(braid_App         app,
//              braid_Vector      ustop,     /**< input, u vector at *tstop* */
//              braid_Vector      u,         /**< input, u vector at *tstart* */
//              braid_Vector      ustop_bar, /**< input / output, adjoint vector for ustop */
//              braid_Vector      u_bar,     /**< input / output, adjoint vector for u */
//              braid_StepStatus  status)
// {
//     int    ts;
//     double tstart, tstop;
//     double deltaT;

//     int nexamples = app->nexamples;
   
//     /* Get the time-step size and current time index*/
//     braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
//     braid_StepStatusGetTIndex(status, &ts);
//     deltaT = tstop - tstart;


//     /* apply the layer backwards for all examples */
//     for (int iex = 0; iex < nexamples; iex++)
//     {

//         if (ts == 0)
//         {
//             app->network->layers[0]->setExample(app->examples[iex]);
//         }
//         else
//         {
//             app->network->layers[ts]->setDt(deltaT);
//         }

//         /* Apply the layer backwards */
//         app->network->layers[ts]->applyBWD(u->state[iex], u_bar->state[iex]);
//     }

//     return 0;
// }


// int 
// my_ResetGradient(braid_App app)
// {
//     int nlayers = app->network->getnLayers();

//     /* Reset bar variables of weights and bias at all layers */
//     for (int ilayer = 0; ilayer < nlayers; ilayer++)
//     {
//         app->network->layers[ilayer]->resetBar();
//     }

//     return 0;
// }




int 
my_Step_Adj(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     fstop,
            braid_Vector     u,
            braid_StepStatus status)
{
    int    ts_start, ts_stop;
    double tstart, tstop;
    double deltaT;
    int    finegrid  = 0;
    int    primaltimestep;
    braid_BaseVector uprimal;
    int    nexamples = app->nexamples;
   
    /* Get the time-step size and current time index*/
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    ts_start       = GetTimeStepIndex(app, tstart); 
    ts_stop        = GetTimeStepIndex(app, tstop); 
    deltaT         = tstop - tstart;
    primaltimestep = GetPrimalIndex(app, ts_stop); 


    /* Get the layer from the primal core */
    _braid_UGetVectorRef(app->primalcore, finegrid, primaltimestep, &uprimal);
    Layer* layer = uprimal->userVector->layer;


    /* Take one step backwards */
    layer->setDt(deltaT);
    for (int iex = 0; iex < nexamples; iex++)
    {
        if (app->examples !=NULL) layer->setExample(app->examples[iex]);

        layer->applyBWD(uprimal->userVector->state[iex], u->state[iex]); // this updates the weights_bar in u->primal_vec->layer
    }

    printf("%d: step %d->%d using layer %d,%1.14e, primal %1.14e, grad[0] %1.14e, %d\n", app->myid, ts_start, ts_stop, layer->getIndex(), layer->getWeights()[3], uprimal->userVector->state[1][1], layer->getWeightsBar()[0], layer->getnDesign());

    /* Add costfunction derivative */
    layer->evalTikh_diff(1.0);

    /* TODO: Add derivative of DDT regularization */

    /* no refinement */
    braid_StepStatusSetRFactor(status, 1);

    return 0;
}

int
my_Init_Adj(braid_App     app,
            double        t,
            braid_Vector *u_ptr)
{
    braid_BaseVector uprimal;
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
    double *aux     = new double[nchannels];

    int finegrid       = 0;
    int ilayer         = GetTimeStepIndex(app, t);
    int primaltimestep = GetPrimalIndex(app, ilayer);


    printf("%d: Init %d (primaltimestep %d)\n", app->myid, ilayer, primaltimestep);

    /* Allocate the adjoint vector and set to zero */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new double*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new double[nchannels];
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
        double** primalstate = uprimal->userVector->state;
        Layer* layer = uprimal->userVector->layer;

        /* Recompute the Classification */
        for (int iex = 0; iex < app->nexamples; iex++)
        {
            /* Copy values into auxiliary vector */
            for (int ic = 0; ic < nchannels; ic++)
            {
                aux[ic] = primalstate[iex][ic];
            }
            /* Apply classification on aux */
            layer->applyFWD(aux);
        }
        
        /* Compute derivative  */
        double loss_bar = 1./app->nexamples; 
        for (int iex = 0; iex < nexamples; iex++)
        {
            layer->evalLoss_diff(aux, u->state[iex], app->labels[iex], loss_bar);

            layer->applyBWD(primalstate[iex], u->state[iex]);
        }
        printf("%d: BWD Loss at %d, using primal %1.14e, adj %1.14e, grad[0] %1.14e\n", app->myid, layer->getIndex(), primalstate[1][1], u->state[9][6], layer->getWeightsBar()[0]);
 

        /* Derivative of tikhonov regularization */
        layer->evalTikh_diff(1.0);

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
    int nexamples = app->nexamples;
   
    *size_ptr = nchannels*nexamples*sizeof(double);
    return 0;
}



int
my_BufPack_Adj(braid_App           app,
               braid_Vector        u,
               void               *buffer,
               braid_BufferStatus  bstatus)
{
    int size;
    double *dbuffer   = (double*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
    
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
    size = nchannels*nexamples*sizeof(double);

    braid_BufferStatusSetSize( bstatus, size);
 
   return 0;
}



int
my_BufUnpack_Adj(braid_App           app,
                 void               *buffer,
                 braid_Vector       *u_ptr,
                 braid_BufferStatus  bstatus)
{

    double *dbuffer   = (double*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
    
    //  /* Allocate the vector */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new double*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new double[nchannels];
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

