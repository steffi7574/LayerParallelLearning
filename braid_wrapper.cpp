#include "braid_wrapper.hpp"
// #include "codi.hpp"
// #include "lib.hpp"


int 
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
    int    ts;
    double tstart, tstop;
    double deltaT;

    int nexamples = app->nexamples;
   
    /* Get the time-step size and current time index*/
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    braid_StepStatusGetTIndex(status, &ts);
    deltaT = tstop - tstart;

    /* apply the layer for all examples */
    for (int iex = 0; iex < nexamples; iex++)
    {
        if (ts == 0)
        {
            app->network->layers[0]->setExample(app->examples[iex]);
        }
        else
        {
            app->network->layers[ts]->setDt(deltaT);
        }

        /* Apply the layer */
        app->network->layers[ts]->applyFWD(u->state[iex]);
    }

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


    /* Allocate the vector */
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
   
    *size_ptr = nchannels*nexamples*sizeof(double);
    return 0;
}



int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{
    double *dbuffer   = (double*) buffer;
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
    
    for (int iex = 0; iex < nexamples; iex++)
    {
        for (int ic = 0; ic < nchannels; ic++)
        {
           dbuffer[iex * nchannels + ic] = (u->state)[iex][ic];
        }
    }

    braid_BufferStatusSetSize( bstatus,  nchannels*nexamples*sizeof(double));
 
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
    
    //  /* Allocate the vector */
    my_Vector* u = (my_Vector *) malloc(sizeof(my_Vector));
    u->state = new double*[nexamples];
    for (int iex = 0; iex < nexamples; iex++)
    {
        u->state[iex] = new double[nchannels];
    }

    /* Unpack the buffer */
    for (int iex = 0; iex < nexamples; iex++)
    {
        for (int ic = 0; ic < nchannels; ic++)
        {
           u->state[iex][ic] = dbuffer[iex * nchannels + ic];
        }
    }
 
    *u_ptr = u;
    return 0;
}


int 
my_ObjectiveT(braid_App              app,
              braid_Vector           u,
              braid_ObjectiveStatus  ostatus,
              double                *objective_ptr)
{
    int    class_id = 0;
    int    success = 0;
    double regul_tik = 0.0;
    double regul_ddt = 0.0;
    double loss      = 0.0;
    double accuracy  = 0.0;

    int nlayers   = app->network->getnLayers();
    int nexamples = app->nexamples;

    /* Get the time index*/
    int ts;
    braid_ObjectiveStatusGetTIndex(ostatus, &ts);
    int ilayer = ts - 1;
    if (ilayer < 0) 
    {
       *objective_ptr = 0.0;
        return 0;
    }

    /* Tikhonov regularization */
    regul_tik = app->network->layers[ilayer]->evalTikh();

    /* ddt-regularization term */
    if (ilayer > 1 && ilayer < nlayers - 1) 
    {
        regul_ddt = app->network->evalRegulDDT(app->network->layers[ilayer-1], app->network->layers[ilayer]);
    }

 
    /* Evaluate loss and accuracy at last layer */
    if (ilayer == nlayers - 1)
    {
        success = 0;
        loss    = 0.0;
        for (int iex = 0; iex < nexamples; iex++)
        {
            loss += app->network->layers[nlayers-1]->evalLoss(u->state[iex], app->labels[iex]);

            /* Test for successful prediction */
            class_id = app->network->layers[nlayers-1]->prediction(u->state[iex]);
            if ( app->labels[iex][class_id] > 0.99 )  
            {
                success++;
            }
        }
        loss     = 1. / nexamples * loss;
        accuracy  = 100.0 * (double) success / nexamples;

        /* Report to app */
        app->loss     = loss;
        app->accuracy = accuracy;
    }


    /* Compute objective function */
    *objective_ptr = loss + app->gamma_tik * regul_tik + app->gamma_ddt * regul_ddt;

    
    return 0;
}


int
my_ObjectiveT_diff(braid_App            app,
                  braid_Vector          u,
                  braid_Vector          u_bar,
                  braid_Real            f_bar,
                  braid_ObjectiveStatus ostatus)
{
    double loss_bar, regul_tik_bar, regul_ddt_bar;
    int nlayers   = app->network->getnLayers();
    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;

    /* Get the time index*/
    int ts;
    braid_ObjectiveStatusGetTIndex(ostatus, &ts);
    int ilayer = ts - 1;
    if (ilayer < 0) 
    {
        return 0;
    }

    /* Derivative of objective function */
    loss_bar      = f_bar;
    regul_tik_bar = f_bar * app->gamma_tik;
    regul_ddt_bar = f_bar * app->gamma_ddt;

    /* Derivative of loss function evaluation */
    if (ilayer == nlayers - 1)
    {
        loss_bar = 1./nexamples * loss_bar;
        for (int iex = 0; iex < nexamples; iex++)
        {
            app->network->layers[nlayers-1]->evalLoss_diff(u->state[iex], u_bar->state[iex], app->labels[iex], loss_bar);
        }
    }
    else /* dfdu = 0.0 */
    {
        for (int iex = 0; iex < nexamples; iex++)
        {
            for (int ic = 0; ic < nchannels; ic++)
            {
                u_bar->state[iex][ic] = 0.0;
            }
        }            
    }

    /* Derivative of ddt-regularization term */
    if (ilayer > 1 && ilayer < nlayers - 1) 
    {
        app->network->evalRegulDDT_diff(app->network->layers[ilayer-1], app->network->layers[ilayer], regul_ddt_bar);
    }

    /* Derivative of the tikhonov regularization term */
    app->network->layers[ilayer]->evalTikh_diff(regul_tik_bar);

    return 0;
}

int
my_Step_diff(braid_App         app,
             braid_Vector      ustop,     /**< input, u vector at *tstop* */
             braid_Vector      u,         /**< input, u vector at *tstart* */
             braid_Vector      ustop_bar, /**< input / output, adjoint vector for ustop */
             braid_Vector      u_bar,     /**< input / output, adjoint vector for u */
             braid_StepStatus  status)
{
    int    ts;
    double tstart, tstop;
    double deltaT;

    int nchannels = app->network->getnChannels();
    int nexamples = app->nexamples;
   
    /* Get the time-step size and current time index*/
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    braid_StepStatusGetTIndex(status, &ts);
    deltaT = tstop - tstart;


    /* apply the layer backwards for all examples */
    for (int iex = 0; iex < nexamples; iex++)
    {

        if (ts == 0)
        {
            app->network->layers[0]->setExample(app->examples[iex]);
        }
        else
        {
            app->network->layers[ts]->setDt(deltaT);
        }

        /* Apply the layer backwards */
        app->network->layers[ts]->applyBWD(u->state[iex], u_bar->state[iex]);
    }

    return 0;
}


int 
my_ResetGradient(braid_App app)
{
    int nlayers = app->network->getnLayers();

    /* Reset bar variables of weights and bias at all layers */
    for (int ilayer = 0; ilayer < nlayers; ilayer++)
    {
        app->network->layers[ilayer]->resetBar();
    }

    return 0;
}
