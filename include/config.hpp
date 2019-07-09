#include "defs.hpp"

#include <cstdio>

#pragma once

#define CONFIG_ARG_MAX_BYTES 128

/* Available activation functions */
enum activation { TANH, RELU, SMRELU };

/* Available network types */
enum networkType { DENSE, CONVOLUTIONAL };

/* Available batch types */
enum batchtype { DETERMINISTIC, STOCHASTIC };

/* Available hessian approximation types */
enum hessiantype { BFGS_SERIAL, LBFGS, IDENTITY };

/* Available stepsize selection methods */
enum stepsizetype { FIXED, BACKTRACKINGLS, ONEOVERK };

class Config {

private:
  /* Linked list for reading config options */
  struct config_option {
    struct config_option *prev;
    //  config_option_t prev;
    char key[CONFIG_ARG_MAX_BYTES];
    char value[CONFIG_ARG_MAX_BYTES];
  };

  /* Helper function: Parse the config file */
  config_option *parsefile(char *path);

public: /* List all configuration options here */
  /* Data set */
  const char *datafolder;
  const char *ftrain_ex;
  const char *ftrain_labels;
  const char *fval_ex;
  const char *fval_labels;
  const char *weightsopenfile;
  const char *weightsclassificationfile;

  int ntraining;
  int nvalidation;
  int nfeatures;
  int nclasses;

  /* Neural Network */
  int nchannels;
  int nlayers;
  MyReal T;
  int activation;
  int network_type;
  int openlayer_type;
  MyReal weights_open_init;
  MyReal weights_init;
  MyReal weights_class_init;

  /* XBraid */
  int braid_cfactor0;
  int braid_cfactor;
  int braid_maxlevels;
  int braid_mincoarse;
  int braid_maxiter;
  MyReal braid_abstol;
  MyReal braid_abstoladj;
  int braid_printlevel;
  int braid_accesslevel;
  int braid_setskip;
  int braid_fmg;
  int braid_nrelax;
  int braid_nrelax0;

  /* Optimization */
  int batch_type;
  int nbatch;
  MyReal gamma_tik;
  MyReal gamma_ddt;
  MyReal gamma_class;
  int stepsize_type;
  MyReal stepsize_init;
  int maxoptimiter;
  MyReal gtol;
  int ls_maxiter;
  MyReal ls_factor;
  int hessianapprox_type;
  int lbfgs_stages;
  int validationlevel;

  /* Constructor sets default values */
  Config();

  /* Destructor */
  ~Config();

  /* Reads the config options from file */
  int readFromFile(char *configfilename);

  /* Writes config options to the file (File must be open!) */
  int writeToFile(FILE *outfile);

  /* Returns a stepsize, depending on the selected stepsize type and current
   * optimization iteration */
  MyReal getStepsize(int optimiter);
};
