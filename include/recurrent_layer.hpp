#include "layer.hpp"

#include <vector>

#pragma once

class Config;

/**
 * Abstract base class for the network layers
 * Subclasses implement
 *    - applyFWD: Forward propagation of data
 *    - applyBWD: Backward propagation of data
 */
class RecurrentLayer : public Layer {
  std::vector<Layer*> layers;
 public:

  RecurrentLayer();
  RecurrentLayer(int idx, int nrecur_sz, int dimI, int dimO, MyReal deltaT, int Activ,
                 MyReal gammatik, MyReal gammaddt);

  virtual ~RecurrentLayer();

  /**
   * Forward propagation of an example
   * In/Out: vector holding the current propagated example
   */
  virtual void applyFWD(MyReal *state);

  /**
   * Backward propagation of an example
   * In:     data     - current example data
   * In/Out: data_bar - adjoint example data that is to be propagated backwards
   * In:     compute_gradient - flag to determin if gradient should be computed
   * (i.e. if weights_bar,bias_bar should be updated or not. In general, update
   * is only done on the finest layer-grid.)
   */
  virtual void applyBWD(MyReal *state, MyReal *state_bar,
                        int compute_gradient);

  void updateSubLayerMemory();

};
