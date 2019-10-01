#include "recurrent_layer.hpp"

#include "config.hpp"

#include <iostream>

RecurrentLayer::RecurrentLayer()
{
}

RecurrentLayer::RecurrentLayer(int idx, int nrecur_sz, int dimI, int dimO, MyReal deltaT, int Activ,
                       MyReal gammatik, MyReal gammaddt)
    : Layer(idx, Layer::RECURRENT, dimI, dimO, 0, 0, deltaT, Activ, gammatik, gammaddt)
{
  nrecur_size = nrecur_sz;

  layers.resize(nrecur_size,0);
  for(int i=0;i<nrecur_size;i++) {
    layers[i] = new DenseLayer(i, dimI, dimO, deltaT/nrecur_size, Activ,gammatik,gammaddt);
    ndesign += layers[i]->getnDesign();
  }

  // printf("RECURRENT SIZE = %d, W = %d, SW = %d\n", nrecur_size,ndesign,layers[0]->getnDesign());

  nweights = ndesign;
}

RecurrentLayer::~RecurrentLayer() 
{
  for(int i=0;i<nrecur_size;i++)
    delete layers[i];

  layers.clear();
}

void RecurrentLayer::applyFWD(MyReal *state)
{
  // set memory for all the sublayers
  updateSubLayerMemory();

  for(int i=0;i<nrecur_size;i++)
    layers[i]->applyFWD(state);
}

void RecurrentLayer::applyBWD(MyReal *state, MyReal *state_bar,
                              int compute_gradient)
{
  // set memory for all the sublayers
  updateSubLayerMemory();

  // build up state vector for the recurrence
  std::vector<MyReal*> state_vecs(nrecur_size,0);
  state_vecs[0] = state;
  for(int i=0;i<nrecur_size-1;i++) {
    MyReal * cur_state = new MyReal[dim_In];
    memcpy(cur_state,state_vecs[i],sizeof(MyReal)*dim_In);

    layers[i]->applyFWD(cur_state);

    state_vecs[i+1] = cur_state;
  }

  // do adjoint solve
  for(int i=nrecur_size-1;i>=0;i--) {
    layers[i]->applyBWD(state_vecs[i],state_bar,compute_gradient);
  }

  // clean up
  for(int i=0;i<nrecur_size-1;i++)
    delete [] state_vecs[i+1];
}

void RecurrentLayer::updateSubLayerMemory()
{
  MyReal * design   = weights;
  MyReal * gradient = weights_bar;

  for(int i=0;i<nrecur_size;i++) {
    layers[i]->setMemory(design,gradient);
    
    design += layers[i]->getnDesign();
    gradient += layers[i]->getnDesign();
  }
}
