// Copyright
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Underlying paper:
//
// Layer-Parallel Training of Deep Residual Neural Networks
// S. Guenther, L. Ruthotto, J.B. Schroder, E.C. Cyr, and N.R. Gauger
//
// Download: https://arxiv.org/pdf/1812.04352.pdf
//
#include "layer.hpp"
#include <assert.h>
#include <math.h>

#include <iostream>

Layer::Layer() {
  dim_In = 0;
  dim_Out = 0;
  dim_Bias = 0;
  ndesign = 0;
  nweights = 0;
  nconv = 0;
  csize = 0;

  index = 0;
  dt = 0.0;
  activ = -1;
  weights = NULL;
  weights_bar = NULL;
  bias = NULL;
  bias_bar = NULL;
  gamma_tik = 0.0;
  gamma_ddt = 0.0;
  update = NULL;
  update_bar = NULL;
}

Layer::Layer(int idx, int Type, int dimI, int dimO, int dimB, int dimW,
             MyReal deltaT, int Activ, MyReal gammatik, MyReal gammaddt)
    : Layer() {
  index = idx;
  type = Type;
  dim_In = dimI;
  dim_Out = dimO;
  dim_Bias = dimB;
  ndesign = dimW + dimB;
  nweights = dimW;
  dt = deltaT;
  activ = Activ;
  gamma_tik = gammatik;
  gamma_ddt = gammaddt;

  update = new MyReal[dimO];
  update_bar = new MyReal[dimO];
}

Layer::~Layer() {
  delete[] update;
  delete[] update_bar;
}

void Layer::setDt(MyReal DT) { dt = DT; }

MyReal Layer::getDt() { return dt; }

void Layer::setMemory(MyReal *design_memloc, MyReal *gradient_memloc) {
  /* Set design and gradient memory locations */
  weights = design_memloc;
  weights_bar = gradient_memloc;

  /* Bias memory locations is a shift by number of weights */
  bias = design_memloc + nweights;
  bias_bar = gradient_memloc + nweights;
}

MyReal Layer::getGammaTik() { return gamma_tik; }

MyReal Layer::getGammaDDT() { return gamma_ddt; }

int Layer::getActivation() { return activ; }

int Layer::getType() { return type; }

MyReal *Layer::getWeights() { return weights; }
MyReal *Layer::getBias() { return bias; }

MyReal *Layer::getWeightsBar() { return weights_bar; }
MyReal *Layer::getBiasBar() { return bias_bar; }

int Layer::getDimIn() { return dim_In; }
int Layer::getDimOut() { return dim_Out; }
int Layer::getDimBias() { return dim_Bias; }
int Layer::getnWeights() { return nweights; }
int Layer::getnDesign() { return ndesign; }

int Layer::getnConv() { return nconv; }
int Layer::getCSize() { return csize; }

int Layer::getIndex() { return index; }

void Layer::print_data(MyReal *data) {
  printf("DATA: ");
  for (int io = 0; io < dim_Out; io++) {
    printf("%1.14e ", data[io]);
  }
  printf("\n");
}

MyReal Layer::activation(MyReal x) {
  MyReal y;
  switch (activ) {
    case TANH:
      y = Layer::tanh_act(x);
      break;
    case RELU:
      y = Layer::ReLu_act(x);
      break;
    case SMRELU:
      y = Layer::SmoothReLu_act(x);
      break;
    default:
      y = -1000000.0;
      printf("ERROR: You should specify an activation function!\n");
      printf("GO HOME AND GET SOME SLEEP!");
      break;
  }
  return y;
}

MyReal Layer::dactivation(MyReal x) {
  MyReal y;
  switch (activ) {
    case TANH:
      y = Layer::dtanh_act(x);
      break;
    case RELU:
      y = Layer::dReLu_act(x);
      break;
    case SMRELU:
      y = Layer::dSmoothReLu_act(x);
      break;
    default:
      y = -1000000.0;
      printf("ERROR: You should specify an activation function!\n");
      printf("GO HOME AND GET SOME SLEEP!");
      break;
  }
  return y;
}

void Layer::packDesign(MyReal *buffer, int size) {
  int nweights = getnWeights();
  int nbias = getDimBias();
  int idx = 0;
  for (int i = 0; i < nweights; i++) {
    buffer[idx] = getWeights()[i];
    idx++;
  }
  for (int i = 0; i < nbias; i++) {
    buffer[idx] = getBias()[i];
    idx++;
  }
  /* Set the rest to zero */
  for (int i = idx; i < size; i++) {
    buffer[idx] = 0.0;
    idx++;
  }
}

void Layer::unpackDesign(MyReal *buffer) {
  int nweights = getnWeights();
  int nbias = getDimBias();

  int idx = 0;
  for (int i = 0; i < nweights; i++) {
    getWeights()[i] = buffer[idx];
    idx++;
  }
  for (int i = 0; i < nbias; i++) {
    getBias()[i] = buffer[idx];
    idx++;
  }
}


MyReal Layer::evalTikh() {
  MyReal tik = 0.0;
  for (int i = 0; i < nweights; i++) {
    tik += pow(weights[i], 2);
  }
  for (int i = 0; i < dim_Bias; i++) {
    tik += pow(bias[i], 2);
  }

  return gamma_tik / 2.0 * tik;
}

void Layer::evalTikh_diff(MyReal regul_bar) {
  regul_bar = gamma_tik * regul_bar;

  /* Derivative bias term */
  for (int i = 0; i < dim_Bias; i++) {
    bias_bar[i] += bias[i] * regul_bar;
  }
  for (int i = 0; i < nweights; i++) {
    weights_bar[i] += weights[i] * regul_bar;
  }
}

MyReal Layer::evalRegulDDT(Layer *layer_prev, MyReal deltat) {
  if (layer_prev == NULL) return 0.0;  // this holds for opening layer

  MyReal diff;
  MyReal regul_ddt = 0.0;

  /* Compute ddt-regularization only if dimensions match  */
  /* this excludes first intermediate layer and classification layer. */
  if (layer_prev->getnDesign() == ndesign && layer_prev->getDimIn() == dim_In &&
      layer_prev->getDimOut() == dim_Out &&
      layer_prev->getDimBias() == dim_Bias &&
      layer_prev->getnWeights() == nweights) {
    for (int iw = 0; iw < nweights; iw++) {
      diff = (getWeights()[iw] - layer_prev->getWeights()[iw]) / deltat;
      regul_ddt += pow(diff, 2);
    }
    for (int ib = 0; ib < dim_Bias; ib++) {
      diff = (getBias()[ib] - layer_prev->getBias()[ib]) / deltat;
      regul_ddt += pow(diff, 2);
    }
    regul_ddt = gamma_ddt / 2.0 * regul_ddt;
  }

  return regul_ddt;
}

void Layer::evalRegulDDT_diff(Layer *layer_prev, Layer *layer_next,
                              MyReal deltat) {
  if (layer_prev == NULL) return;
  if (layer_next == NULL) return;

  MyReal diff;
  int regul_bar = gamma_ddt / (deltat * deltat);

  /* Left sided derivative term */
  if (layer_prev->getnDesign() == ndesign && layer_prev->getDimIn() == dim_In &&
      layer_prev->getDimOut() == dim_Out &&
      layer_prev->getDimBias() == dim_Bias &&
      layer_prev->getnWeights() == nweights) {
    for (int ib = 0; ib < dim_Bias; ib++) {
      diff = getBias()[ib] - layer_prev->getBias()[ib];
      getBiasBar()[ib] += diff * regul_bar;
    }

    for (int iw = 0; iw < nweights; iw++) {
      diff = getWeights()[iw] - layer_prev->getWeights()[iw];
      getWeightsBar()[iw] += diff * regul_bar;
    }
  }

  /* Right sided derivative term */
  if (layer_next->getnDesign() == ndesign && layer_next->getDimIn() == dim_In &&
      layer_next->getDimOut() == dim_Out &&
      layer_next->getDimBias() == dim_Bias &&
      layer_next->getnWeights() == nweights) {
    for (int ib = 0; ib < dim_Bias; ib++) {
      diff = getBias()[ib] - layer_next->getBias()[ib];
      getBiasBar()[ib] += diff * regul_bar;
    }

    for (int iw = 0; iw < nweights; iw++) {
      diff = getWeights()[iw] - layer_next->getWeights()[iw];
      getWeightsBar()[iw] += diff * regul_bar;
    }
  }
}

void Layer::setExample(MyReal *example_ptr) {}

void Layer::setLabel(MyReal *example_ptr) {}

DenseLayer::DenseLayer(int idx, int dimI, int dimO, MyReal deltaT, int Activ,
                       MyReal gammatik, MyReal gammaddt)
    : Layer(idx, DENSE, dimI, dimO, 1, dimI * dimO, deltaT, Activ, gammatik,
            gammaddt) {}

DenseLayer::~DenseLayer() {}

void DenseLayer::applyFWD(MyReal *state) {
  /* Affine transformation */
  for (int io = 0; io < dim_Out; io++) {
    /* Apply weights */
    update[io] = vecdot(dim_In, &(weights[io * dim_In]), state);

    /* Add bias */
    update[io] += bias[0];
  }

  /* Apply step */
  for (int io = 0; io < dim_Out; io++) {
    state[io] = state[io] + dt * activation(update[io]);
  }
}

void DenseLayer::applyBWD(MyReal *state, MyReal *state_bar,
                          int compute_gradient) {
  /* state_bar is the adjoint of the state variable, it contains the
     old time adjoint informationk, and is modified on the way out to
     contain the update. */

  /* Derivative of the step */
  for (int io = 0; io < dim_Out; io++) {
    /* Recompute affine transformation */
    update[io] = vecdot(dim_In, &(weights[io * dim_In]), state);
    update[io] += bias[0];

    /* Derivative: This is the update from old time */
    update_bar[io] = dt * dactivation(update[io]) * state_bar[io];
  }

  /* Derivative of linear transformation */
  for (int io = 0; io < dim_Out; io++) {
    /* Derivative of bias addition */
    if (compute_gradient) bias_bar[0] += update_bar[io];

    /* Derivative of weight application */
    for (int ii = 0; ii < dim_In; ii++) {
      if (compute_gradient)
        weights_bar[io * dim_In + ii] += state[ii] * update_bar[io];
      state_bar[ii] += weights[io * dim_In + ii] * update_bar[io];
    }
  }
}

OpenDenseLayer::OpenDenseLayer(int dimI, int dimO, int Activ, MyReal gammatik)
    : DenseLayer(-1, dimI, dimO, 1.0, Activ, gammatik, 0.0) {
  type = OPENDENSE;
  example = NULL;
}

OpenDenseLayer::~OpenDenseLayer() {}

void OpenDenseLayer::setExample(MyReal *example_ptr) { example = example_ptr; }

void OpenDenseLayer::applyFWD(MyReal *state) {
  /* affine transformation */
  for (int io = 0; io < dim_Out; io++) {
    /* Apply weights */
    update[io] = vecdot(dim_In, &(weights[io * dim_In]), example);

    /* Add bias */
    update[io] += bias[0];
  }

  /* Step */
  for (int io = 0; io < dim_Out; io++) {
    state[io] = activation(update[io]);
  }
}

void OpenDenseLayer::applyBWD(MyReal *state, MyReal *state_bar,
                              int compute_gradient) {
  /* Derivative of step */
  for (int io = 0; io < dim_Out; io++) {
    /* Recompute affine transformation */
    update[io] = vecdot(dim_In, &(weights[io * dim_In]), example);
    update[io] += bias[0];

    /* Derivative */
    update_bar[io] = dactivation(update[io]) * state_bar[io];
    state_bar[io] = 0.0;
  }

  /* Derivative of affine transformation */
  if (compute_gradient) {
    for (int io = 0; io < dim_Out; io++) {
      /* Derivative of bias addition */
      bias_bar[0] += update_bar[io];

      /* Derivative of weight application */
      for (int ii = 0; ii < dim_In; ii++) {
        weights_bar[io * dim_In + ii] += example[ii] * update_bar[io];
      }
    }
  }
}

OpenExpandZero::OpenExpandZero(int dimI, int dimO)
    : Layer(-1, OPENZERO, dimI, dimO, 0, 0, 1.0, -1, 0.0, 0.0) {
  /* this layer doesn't have any design variables. */
  ndesign = 0;
  nweights = 0;
}

OpenExpandZero::~OpenExpandZero() {}

void OpenExpandZero::setExample(MyReal *example_ptr) { example = example_ptr; }

void OpenExpandZero::applyFWD(MyReal *state) {
  for (int ii = 0; ii < dim_In; ii++) {
    state[ii] = example[ii];
  }
  for (int io = dim_In; io < dim_Out; io++) {
    state[io] = 0.0;
  }
}

void OpenExpandZero::applyBWD(MyReal *state, MyReal *state_bar,
                              int compute_gradient) {
  for (int ii = 0; ii < dim_Out; ii++) {
    state_bar[ii] = 0.0;
  }
}

OpenConvLayer::OpenConvLayer(int dimI, int dimO)
    : Layer(-1, OPENCONV, dimI, dimO, 0, 0, 1.0, -1, 0.0, 0.0) {
  /* this layer doesn't have any design variables. */
  ndesign = 0;
  nweights = 0;
  dim_Bias = 0;

  nconv = dim_Out / dim_In;

  assert(nconv * dim_In == dim_Out);
}

OpenConvLayer::~OpenConvLayer() {}

void OpenConvLayer::setExample(MyReal *example_ptr) { example = example_ptr; }

void OpenConvLayer::applyFWD(MyReal *state) {
  // replicate the image data
  for (int img = 0; img < nconv; img++) {
    for (int ii = 0; ii < dim_In; ii++) {
      state[ii + dim_In * img] = example[ii];
    }
  }
}

void OpenConvLayer::applyBWD(MyReal *state, MyReal *state_bar,
                             int compute_gradient) {
  for (int ii = 0; ii < dim_Out; ii++) {
    state_bar[ii] = 0.0;
  }
}

OpenConvLayerMNIST::OpenConvLayerMNIST(int dimI, int dimO)
    : OpenConvLayer(dimI, dimO) {
  type = OPENCONVMNIST;
}

OpenConvLayerMNIST::~OpenConvLayerMNIST() {}

void OpenConvLayerMNIST::applyFWD(MyReal *state) {
  // replicate the image data
  for (int img = 0; img < nconv; img++) {
    for (int ii = 0; ii < dim_In; ii++) {
      // The MNIST data is integer from [0, 255], so we rescale it to floats
      // over the range[0,6]
      //
      // Also, rescale tanh so that it appropriately activates over the x-range
      // of [0,6]
      state[ii + dim_In * img] = tanh((6.0 * example[ii] / 255.0) - 3.0) + 1;
    }
  }
}

void OpenConvLayerMNIST::applyBWD(MyReal *state, MyReal *state_bar,
                                  int compute_gradient) {
  // Derivative of step
  for (int img = 0; img < nconv; img++) {
    for (int ii = 0; ii < dim_In; ii++) {
      state_bar[ii + dim_In * img] =
          (1.0 - pow(tanh(example[ii]), 2)) * state_bar[ii + dim_In * img];
      // state_bar[ii + dim_In*img] = 0.0;
    }
  }

  // Derivative of affine transformation
  // This is "0" because we have no bias or weights
}

ClassificationLayer::ClassificationLayer(int idx, int dimI, int dimO,
                                         MyReal gammatik)
    : Layer(idx, CLASSIFICATION, dimI, dimO, dimO, dimI * dimO, 1.0, -1, 0.0,
            0.0) {
  gamma_tik = gammatik;
  /* Allocate the probability vector */
  probability = new MyReal[dimO];
}

ClassificationLayer::~ClassificationLayer() { delete[] probability; }

void ClassificationLayer::setLabel(MyReal *label_ptr) { label = label_ptr; }

void ClassificationLayer::applyFWD(MyReal *state) {
  /* Compute affine transformation */
  for (int io = 0; io < dim_Out; io++) {
    /* Apply weights */
    update[io] = vecdot(dim_In, &(weights[io * dim_In]), state);
    /* Add bias */
    update[io] += bias[io];
  }

  /* Data normalization y - max(y) (needed for stable softmax evaluation */
  normalize(update);

  if (dim_In < dim_Out) {
    printf(
        "Error: nchannels < nclasses. Implementation of classification "
        "layer doesn't support this setting. Change! \n");
    exit(1);
  }

  /* Apply step */
  for (int io = 0; io < dim_Out; io++) {
    state[io] = update[io];
  }
  /* Set remaining to zero */
  for (int ii = dim_Out; ii < dim_In; ii++) {
    state[ii] = 0.0;
  }
}

void ClassificationLayer::applyBWD(MyReal *state, MyReal *state_bar,
                                   int compute_gradient) {
  /* Recompute affine transformation */
  for (int io = 0; io < dim_Out; io++) {
    update[io] = vecdot(dim_In, &(weights[io * dim_In]), state);
    update[io] += bias[io];
  }

  /* Derivative of step */
  for (int ii = dim_Out; ii < dim_In; ii++) {
    state_bar[ii] = 0.0;
  }
  for (int io = 0; io < dim_Out; io++) {
    update_bar[io] = state_bar[io];
    state_bar[io] = 0.0;
  }

  /* Derivative of the normalization */
  normalize_diff(update, update_bar);

  /* Derivatie of affine transformation */
  for (int io = 0; io < dim_Out; io++) {
    /* Derivative of bias addition */
    if (compute_gradient) bias_bar[io] += update_bar[io];

    /* Derivative of weight application */
    for (int ii = 0; ii < dim_In; ii++) {
      if (compute_gradient)
        weights_bar[io * dim_In + ii] += state[ii] * update_bar[io];
      state_bar[ii] += weights[io * dim_In + ii] * update_bar[io];
    }
  }
}

void ClassificationLayer::normalize(MyReal *data) {
  /* Find maximum value */
  MyReal max = vecmax(dim_Out, data);
  /* Shift the data vector */
  for (int io = 0; io < dim_Out; io++) {
    data[io] = data[io] - max;
  }
}

void ClassificationLayer::normalize_diff(MyReal *data, MyReal *data_bar) {
  MyReal max_b = 0.0;
  /* Derivative of the shift */
  for (int io = 0; io < dim_Out; io++) {
    max_b -= data_bar[io];
  }
  /* Derivative of the vecmax */
  int i_max = argvecmax(dim_Out, data);
  data_bar[i_max] += max_b;
}

MyReal ClassificationLayer::crossEntropy(MyReal *data_Out) {
  MyReal label_pr, exp_sum;
  MyReal CELoss;

  /* Label projection */
  label_pr = vecdot(dim_Out, label, data_Out);

  /* Compute sum_i (exp(x_i)) */
  exp_sum = 0.0;
  for (int io = 0; io < dim_Out; io++) {
    exp_sum += exp(data_Out[io]);
  }

  /* Cross entropy loss function */
  CELoss = -label_pr + log(exp_sum);

  return CELoss;
}

void ClassificationLayer::crossEntropy_diff(MyReal *data_Out,
                                            MyReal *data_Out_bar,
                                            MyReal loss_bar) {
  MyReal exp_sum, exp_sum_bar;
  MyReal label_pr_bar = -loss_bar;

  /* Recompute exp_sum */
  exp_sum = 0.0;
  for (int io = 0; io < dim_Out; io++) {
    exp_sum += exp(data_Out[io]);
  }

  /* derivative of log(exp_sum) */
  exp_sum_bar = 1. / exp_sum * loss_bar;
  for (int io = 0; io < dim_Out; io++) {
    data_Out_bar[io] = exp(data_Out[io]) * exp_sum_bar;
  }

  /* Derivative of vecdot */
  for (int io = 0; io < dim_Out; io++) {
    data_Out_bar[io] += label[io] * label_pr_bar;
  }
}

int ClassificationLayer::prediction(MyReal *data_Out, int *class_id_ptr) {
  MyReal exp_sum, max;
  int class_id = -1;
  int success = 0;

  /* Compute sum_i (exp(x_i)) */
  max = -1.0;
  exp_sum = 0.0;
  for (int io = 0; io < dim_Out; io++) {
    exp_sum += exp(data_Out[io]);
  }

  for (int io = 0; io < dim_Out; io++) {
    /* Compute class probabilities (Softmax) */
    probability[io] = exp(data_Out[io]) / exp_sum;

    /* Predicted class is the one with maximum probability */
    if (probability[io] > max) {
      max = probability[io];
      class_id = io;
    }
  }

  /* Test for successful prediction */
  if (label[class_id] > 0.99) {
    success = 1;
  }

  /* return */
  *class_id_ptr = class_id;
  return success;
}

MyReal Layer::ReLu_act(MyReal x) {
  MyReal max = 0.0;

  if (x > 0.0) max = x;

  return max;
}

MyReal Layer::dReLu_act(MyReal x) {
  MyReal diff;
  if (x >= 0.0)
    diff = 1.0;
  else
    diff = 0.0;

  return diff;
}

MyReal Layer::SmoothReLu_act(MyReal x) {
  /* range of quadratic interpolation */
  MyReal eta = 0.1;
  /* Coefficients of quadratic interpolation */
  MyReal a = 1. / (4. * eta);
  MyReal b = 1. / 2.;
  MyReal c = eta / 4.;

  if (-eta < x && x < eta) {
    /* Quadratic Activation */
    return a * pow(x, 2) + b * x + c;
  } else {
    /* ReLu Activation */
    return Layer::ReLu_act(x);
  }
}

MyReal Layer::dSmoothReLu_act(MyReal x) {
  /* range of quadratic interpolation */
  MyReal eta = 0.1;
  /* Coefficients of quadratic interpolation */
  MyReal a = 1. / (4. * eta);
  MyReal b = 1. / 2.;

  if (-eta < x && x < eta) {
    return 2. * a * x + b;
  } else {
    return Layer::dReLu_act(x);
  }
}

MyReal Layer::tanh_act(MyReal x) { return tanh(x); }

MyReal Layer::dtanh_act(MyReal x) {
  MyReal diff = 1.0 - pow(tanh(x), 2);

  return diff;
}

ConvLayer::ConvLayer(int idx, int dimI, int dimO, int csize_in, int nconv_in,
                     MyReal deltaT, int Activ, MyReal Gammatik, MyReal Gammaddt)
    : Layer(idx, CONVOLUTION, dimI, dimO, dimI / nconv_in,
            csize_in * csize_in * nconv_in * nconv_in, deltaT, Activ, Gammatik,
            Gammaddt) {
  csize = csize_in;
  nconv = nconv_in;

  fcsize = floor(csize / 2.0);
  csize2 = csize * csize;

  img_size = dim_In / nconv;
  img_size_sqrt = round(sqrt(img_size));

  // nweights = csize*csize*nconv*nconv;
  // ndesign = nweights + dimI/nconv; // must add to account for the bias
}

ConvLayer::~ConvLayer() {}

/**
 * This method is designed to be used only in the applyBWD. It computes the
 * derivative of the objective with respect to the weights. In particular
 * if you objective is $g$ and your kernel operator has value tau at index
 * a,b then
 *
 * d_tau [ g] = \sum_{image j,k} tau state_{j+a,k+b} * update_bar_{j,k}
 *
 * Note that we assume that update_bar is
 *
 *   update_bar = dt * dactivation * state_bar
 *
 * Where state_bar _must_ be at the old time. Note that the adjoint variable
 * state_bar carries withit all the information of the objective derivative.
 */
MyReal ConvLayer::updateWeightDerivative(
    MyReal *state, MyReal *update_bar, int output_conv, /* output convolution */
    int j,                                              /* pixel index */
    int k)                                              /* pixel index */
{
  MyReal val = 0;

  int fcsize_s_l = -fcsize;
  int fcsize_s_u = fcsize;
  int fcsize_t_l = -fcsize;
  int fcsize_t_u = fcsize;
  int fcsize_s_l_adj = -fcsize;
  int fcsize_t_l_adj = -fcsize;

  if ((j + fcsize_s_l) < 0) fcsize_s_l = -j;
  if ((k + fcsize_t_l) < 0) fcsize_t_l = -k;
  if ((j + fcsize_s_u) >= img_size_sqrt) fcsize_s_u = img_size_sqrt - j - 1;
  if ((k + fcsize_t_u) >= img_size_sqrt) fcsize_t_u = img_size_sqrt - k - 1;

  if ((j - fcsize_s_l_adj) >= img_size_sqrt)
    fcsize_s_l_adj = -(img_size_sqrt - j - 1);
  if ((k - fcsize_t_l_adj) >= img_size_sqrt)
    fcsize_t_l_adj = -(img_size_sqrt - k - 1);

  const int fcsize_s = fcsize_s_u - fcsize_s_l;
  const int fcsize_t = fcsize_t_u - fcsize_t_l;

  int center_index = j * img_size_sqrt + k;
  int input_wght_idx = output_conv * csize2 * nconv + fcsize * (csize + 1);

  int offset = fcsize_t_l + img_size_sqrt * fcsize_s_l;
  int wght_idx = fcsize_t_l + csize * fcsize_s_l;

  int offset_adj = -fcsize_t_l_adj - img_size_sqrt * fcsize_s_l_adj;
  int wght_idx_adj = fcsize_t_l_adj + csize * fcsize_s_l_adj;

  for (int input_image = 0; input_image < nconv;
       input_image++, center_index += img_size, input_wght_idx += csize2) {
    MyReal update_val = update_bar[center_index];

    MyReal *state_base = state + center_index + offset;
    MyReal *weights_bar_base = weights_bar + input_wght_idx + wght_idx;

    MyReal *update_base = update_bar + center_index + offset_adj;
    MyReal *weights_base = weights + input_wght_idx + wght_idx_adj;

    // weight derivative
    for (int s = 0; s <= fcsize_s; s++, state_base += img_size_sqrt,
             weights_bar_base += csize, update_base -= img_size_sqrt,
             weights_base += csize) {
      MyReal *state_local = state_base;
      MyReal *weights_bar_local = weights_bar_base;

      MyReal *update_local = update_base;
      MyReal *weights_local = weights_base;

      for (int t = 0; t <= fcsize_t; t++, state_local++, weights_bar_local++,
               update_local--, weights_local++) {
        (*weights_bar_local) += update_val * (*state_local);
        val += (*update_local) * (*weights_local);
      }
    }
  }

  return val;
}

MyReal ConvLayer::apply_conv(MyReal *state,
                             int output_conv, /* output convolution */
                             int j,           /* pixel index */
                             int k)           /* pixel index */
{
  MyReal val = 0.0;

  int fcsize_s_l = -fcsize;
  int fcsize_s_u = fcsize;
  int fcsize_t_l = -fcsize;
  int fcsize_t_u = fcsize;

  // protect indexing at image boundaries
  if ((j + fcsize_s_l) < 0) fcsize_s_l = -j;
  if ((k + fcsize_t_l) < 0) fcsize_t_l = -k;
  if ((j + fcsize_s_u) >= img_size_sqrt) fcsize_s_u = img_size_sqrt - j - 1;
  if ((k + fcsize_t_u) >= img_size_sqrt) fcsize_t_u = img_size_sqrt - k - 1;

  const int fcsize_s = fcsize_s_u - fcsize_s_l;
  const int fcsize_t = fcsize_t_u - fcsize_t_l;

  int center_index =
      j * img_size_sqrt + k + fcsize_t_l + img_size_sqrt * fcsize_s_l;
  int input_wght_idx = output_conv * csize2 * nconv + fcsize * (csize + 1) +
                       fcsize_t_l + csize * fcsize_s_l;

  /* loop over all the images */
  for (int input_image = 0; input_image < nconv;
       input_image++, center_index += img_size, input_wght_idx += csize2) {
    MyReal *state_base = state + center_index;
    MyReal *weights_base = weights + input_wght_idx;

    for (int s = 0; s <= fcsize_s;
         s++, state_base += img_size_sqrt, weights_base += csize) {
      MyReal *state_local = state_base;
      MyReal *weights_local = weights_base;

      for (int t = 0; t <= fcsize_t; t++, state_local++, weights_local++) {
        val += (*state_local) * (*weights_local);
      }
    }
  }

  return val;
}

MyReal ConvLayer::apply_conv_trans(MyReal *state,
                                   int output_conv, /* output convolution */
                                   int j,           /* pixel index */
                                   int k)           /* pixel index */
{
  MyReal val = 0.0;

  int fcsize_s_l = -fcsize;
  int fcsize_s_u = fcsize;
  int fcsize_t_l = -fcsize;
  int fcsize_t_u = fcsize;

  if ((j - fcsize_s_u) < 0) fcsize_s_u = j;
  if ((k - fcsize_t_u) < 0) fcsize_t_u = k;
  if ((j - fcsize_s_l) >= img_size_sqrt) fcsize_s_l = -(img_size_sqrt - j - 1);
  if ((k - fcsize_t_l) >= img_size_sqrt) fcsize_t_l = -(img_size_sqrt - k - 1);

  const int fcsize_s = fcsize_s_u - fcsize_s_l;
  const int fcsize_t = fcsize_t_u - fcsize_t_l;

  /* loop over all the images */
  int center_index = j * img_size_sqrt + k;
  int input_wght_idx = output_conv * csize2 * nconv;
  for (int input_image = 0; input_image < nconv;
       input_image++, center_index += img_size, input_wght_idx += csize2) {
    int offset = center_index - fcsize_t_l;
    int wght_idx = input_wght_idx + fcsize * (csize + 1) + fcsize_t_l;

    MyReal *state_base = state + offset - img_size_sqrt * fcsize_s_l;
    MyReal *weights_base = weights + wght_idx + csize * fcsize_s_l;

    for (int s = 0; s <= fcsize_s;
         s++, state_base -= img_size_sqrt, weights_base += csize) {
      MyReal *state_local = state_base;
      MyReal *weights_local = weights_base;

      for (int t = 0; t <= fcsize_t; t++, state_local--, weights_local++) {
        val += (*state_local) * (*weights_local);
      }
    }
  }

  return val;
}

void ConvLayer::applyFWD(MyReal *state) {
  /* Apply step */
  for (int io = 0; io < dim_Out; io++) update[io] = state[io];

  /* Affine transformation */
  for (int i = 0; i < nconv; i++) {
    for (int j = 0; j < img_size_sqrt; j++) {
      int state_index = i * img_size + j * img_size_sqrt;
      MyReal *update_local = state + state_index;
      MyReal *bias_local = bias + j * img_size_sqrt;

      for (int k = 0; k < img_size_sqrt; k++, update_local++, bias_local++) {
        // (*update_local) += dt*tanh(apply_conv(update, i, j, k) +
        // (*bias_local));
        (*update_local) +=
            dt * ReLu_act(apply_conv(update, i, j, k) + (*bias_local));
      }
    }
  }
}

void ConvLayer::applyBWD(MyReal *state, MyReal *state_bar,
                         int compute_gradient) {
  /* state_bar is the adjoint of the state variable, it contains the
     old time adjoint information, and is modified on the way out to
     contain the update. */

  /* Okay, for my own clarity:
     state       = forward state solution
     state_bar   = backward adjoint solution (in - new time, out - current time)
     update_bar  = update to the bacward solution, this is "MyReal dipped" in
     that it is used to compute the weight and bias derivative. Note that
     because this is written as a forward update (the residual is F = u_{n+1} -
     u_n - dt * sigma(W_n * u_n + b_n) the adjoint variable is also the
     derivative of the objective with respect to the solution. weights_bar =
     Derivative of the objective with respect to the weights bias_bar    =
     Derivative of the objective with respect to the bias


     More details: Assume that the objective is 'g', and the constraint in
     residual form is F(u,W). Then

       d_{W_n} g = \partial_{u} g * \partial_{W_n} u

     Note that $\partial_{u} g$ only depends on the final layer. Expanding
     around the constraint then gives

       d_{W_n} g = \partial_{u} g * (\partial_{u} F)^{-1} * \partial_{W_n} F

     and now doing the standard adjoint thing we get

       d_{W_n} g = (\partial_{u} F)^{-T} * \partial_{u} g ) * \partial_{W_n} F

     yielding

       d_{W_n} g = state_bar * \partial_{W_n} F

     This is directly

       weights_bar = state_bar * \partial_{W_n} F

     computed below. Similar for the bias.
   */

  /* Affine transformation, and derivative of time step */

  /* loop over number convolutions */
  for (int i = 0; i < nconv; i++) {
    /* loop over full image */
    for (int j = 0; j < img_size_sqrt; j++) {
      int state_index = i * img_size + j * img_size_sqrt;
      MyReal *state_bar_local = state_bar + state_index;
      MyReal *update_bar_local = update_bar + state_index;
      MyReal *bias_local = bias + j * img_size_sqrt;

      for (int k = 0; k < img_size_sqrt;
           k++, state_bar_local++, update_bar_local++, bias_local++) {
        /* compute the affine transformation */
        MyReal local_update = apply_conv(state, i, j, k) + (*bias_local);

        /* derivative of the update, this is the contribution from old time */
        // (*update_bar_local) = dt * dactivation(local_update) *
        // (*state_bar_local);
        // (*update_bar_local) = dt * (1.0-pow(tanh(local_update),2)) *
        // (*state_bar_local);
        (*update_bar_local) = dt * dReLu_act(local_update) * (*state_bar_local);
      }
    }
  }

  /* Loop over the output dimensions */
  for (int i = 0; i < nconv; i++) {
    /* loop over full image */
    for (int j = 0; j < img_size_sqrt; j++) {
      int state_index = i * img_size + j * img_size_sqrt;

      MyReal *state_bar_local = state_bar + state_index;
      MyReal *update_bar_local = update_bar + state_index;
      MyReal *bias_bar_local = bias_bar + j * img_size_sqrt;

      for (int k = 0; k < img_size_sqrt;
           k++, state_bar_local++, update_bar_local++, bias_bar_local++) {
        if (compute_gradient) {
          (*bias_bar_local) += (*update_bar_local);

          (*state_bar_local) +=
              updateWeightDerivative(state, update_bar, i, j, k);
        } else {
          (*state_bar_local) += apply_conv_trans(update_bar, i, j, k);
        }
      }
    }

  }  // end for i
}
