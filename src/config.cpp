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
// S. Guenther, L. Ruthotto, J.B. Schroder, E.C. Czr, and N.R. Gauger
//
// Download: https://arxiv.org/pdf/1812.04352.pdf
//
#include "config.hpp"

#include <ctype.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

Config::Config() {
  /* --- Set DEFAULT parameters of the config file options --- */

  /* Data st */
  datafolder = "NONE";
  ftrain_ex = "NONE";
  fval_ex = "NONE";
  ftrain_labels = "NONE";
  fval_labels = "NONE";
  weightsopenfile = "NONE";
  weightsclassificationfile = "NONE";

  ntraining = 5000;
  nvalidation = 200;
  nfeatures = 2;
  nclasses = 5;

  /* Nested Iteration */
  NI_interp_type = 0;
  NI_levels = 1;
  NI_rfactor = 1;
  NI_tols = (int*) malloc(sizeof(int));
  NI_tols[0] = 60; 

  /* Neural Network */
  nchannels = 8;
  nlayers = 32;
  T = 10.0;
  activation = RELU;
  network_type = DENSE;
  openlayer_type = 0;
  weights_open_init = 0.001;
  weights_init = 0.0;
  weights_class_init = 0.001;

  /* XBraid */
  braid_cfactor0 = 4;
  braid_cfactor = 4;
  braid_maxlevels = 10;
  braid_mincoarse = 10;
  braid_maxiter = 3;
  braid_abstol = 1e-10;
  braid_abstoladj = 1e-06;
  braid_printlevel = 1;
  braid_accesslevel = 0;
  braid_setskip = 0;
  braid_fmg = 0;
  braid_nrelax0 = 1;
  braid_nrelax = 1;

  /* Optimization */
  batch_type = DETERMINISTIC;
  nbatch = ntraining;  // full batch
  gamma_tik = 1e-07;
  gamma_ddt = 1e-07;
  gamma_class = 1e-07;
  stepsize_type = BACKTRACKINGLS;
  stepsize_init = 1.0;
  maxoptimiter = 500;
  gtol = 1e-08;
  ls_maxiter = 20;
  ls_factor = 0.5;
  hessianapprox_type = LBFGS;
  lbfgs_stages = 20;
  validationlevel = 1;
}

Config::~Config() { free(NI_tols); }

int Config::readFromFile(char *configfilename) {
   
   /* Tracks number of Nested Iteration Tolerances */
   int NI_num_tols=1;

  /* Parse the config file */
  config_option *co;
  if ((co = parsefile(configfilename)) == NULL) {
    perror("parsefile()");
    return -1;
  }

  /* Set the config options */
  while (1) {
    if (strcmp(co->key, "datafolder") == 0) {
      datafolder = co->value;
    } else if (strcmp(co->key, "ftrain_ex") == 0) {
      ftrain_ex = co->value;
    } else if (strcmp(co->key, "ftrain_labels") == 0) {
      ftrain_labels = co->value;
    } else if (strcmp(co->key, "fval_ex") == 0) {
      fval_ex = co->value;
    } else if (strcmp(co->key, "fval_labels") == 0) {
      fval_labels = co->value;
    } else if (strcmp(co->key, "ntraining") == 0) {
      ntraining = atoi(co->value);
    } else if (strcmp(co->key, "nvalidation") == 0) {
      nvalidation = atoi(co->value);
    } else if (strcmp(co->key, "nfeatures") == 0) {
      nfeatures = atoi(co->value);
    } else if (strcmp(co->key, "nchannels") == 0) {
      nchannels = atoi(co->value);
    } else if (strcmp(co->key, "nclasses") == 0) {
      nclasses = atoi(co->value);
    }
    if (strcmp(co->key, "weightsopenfile") == 0) {
      weightsopenfile = co->value;
    }
    if (strcmp(co->key, "weightsclassificationfile") == 0) {
      weightsclassificationfile = co->value;
    } else if (strcmp(co->key, "nlayers") == 0) {
      nlayers = atoi(co->value);

      if (nlayers < 3) {
        printf(
            "\n\n ERROR: nlayers=%d too small! Choose minimum three layers "
            "(openlayer, one hidden layer, classification layer)!\n\n",
            nlayers);
        return -1;
      }
    } else if (strcmp(co->key, "activation") == 0) {
      if (strcmp(co->value, "tanh") == 0) {
        activation = TANH;
      } else if (strcmp(co->value, "ReLu") == 0) {
        activation = RELU;
      } else if (strcmp(co->value, "SmoothReLu") == 0) {
        activation = SMRELU;
      } else {
        printf("Invalid activation function!");
        return -1;
      }
    } else if (strcmp(co->key, "network_type") == 0) {
      if (strcmp(co->value, "dense") == 0) {
        network_type = DENSE;
      } else if (strcmp(co->value, "convolutional") == 0) {
        network_type = CONVOLUTIONAL;
      } else {
        printf("Invalid network type !");
        return -1;
      }
    } else if (strcmp(co->key, "T") == 0) {
      T = atof(co->value);
    } else if (strcmp(co->key, "NI_levels") == 0) {
      NI_levels = atoi(co->value);
    } else if (strcmp(co->key, "NI_interp_type") == 0) {
      NI_interp_type = atoi(co->value);
    } else if (strcmp(co->key, "NI_rfactor") == 0) {
      NI_rfactor = atoi(co->value);
    } else if (strcmp(co->key, "NI_tols") == 0) {
      free(NI_tols);
      string_to_intarray(co->value, &NI_tols, &NI_num_tols);
    } else if (strcmp(co->key, "braid_cfactor") == 0) {
      braid_cfactor = atoi(co->value);
    } else if (strcmp(co->key, "braid_cfactor0") == 0) {
      braid_cfactor0 = atoi(co->value);
    } else if (strcmp(co->key, "braid_maxlevels") == 0) {
      braid_maxlevels = atoi(co->value);
    } else if (strcmp(co->key, "braid_mincoarse") == 0) {
      braid_mincoarse = atoi(co->value);
    } else if (strcmp(co->key, "braid_maxiter") == 0) {
      braid_maxiter = atoi(co->value);
    } else if (strcmp(co->key, "braid_abstol") == 0) {
      braid_abstol = atof(co->value);
    } else if (strcmp(co->key, "braid_adjtol") == 0) {
      braid_abstoladj = atof(co->value);
    } else if (strcmp(co->key, "braid_printlevel") == 0) {
      braid_printlevel = atoi(co->value);
    } else if (strcmp(co->key, "braid_accesslevel") == 0) {
      braid_accesslevel = atoi(co->value);
    } else if (strcmp(co->key, "braid_setskip") == 0) {
      braid_setskip = atoi(co->value);
    } else if (strcmp(co->key, "braid_fmg") == 0) {
      braid_fmg = atoi(co->value);
    } else if (strcmp(co->key, "braid_nrelax") == 0) {
      braid_nrelax = atoi(co->value);
    } else if (strcmp(co->key, "braid_nrelax0") == 0) {
      braid_nrelax0 = atoi(co->value);
    } else if (strcmp(co->key, "batch_type") == 0) {
      if (strcmp(co->value, "deterministic") == 0) {
        batch_type = DETERMINISTIC;
      } else if (strcmp(co->value, "stochastic") == 0) {
        batch_type = STOCHASTIC;
      } else {
        printf(
            "Invalid optimization type! Should be either 'deterministic' or "
            "'stochastic'!");
        return -1;
      }
    } else if (strcmp(co->key, "nbatch") == 0) {
      nbatch = atoi(co->value);
    } else if (strcmp(co->key, "gamma_tik") == 0) {
      gamma_tik = atof(co->value);
    } else if (strcmp(co->key, "gamma_ddt") == 0) {
      gamma_ddt = atof(co->value);
    } else if (strcmp(co->key, "gamma_class") == 0) {
      gamma_class = atof(co->value);
    } else if (strcmp(co->key, "stepsize_type") == 0) {
      if (strcmp(co->value, "fixed") == 0) {
        stepsize_type = FIXED;
      } else if (strcmp(co->value, "backtrackingLS") == 0) {
        stepsize_type = BACKTRACKINGLS;
      } else if (strcmp(co->value, "oneoverk") == 0) {
        stepsize_type = ONEOVERK;
      } else {
        printf(
            "Invalid stepsize type! Should be either 'fixed' or "
            "'backtrackingLS' or 'oneoverk' !");
        return -1;
      }
    } else if (strcmp(co->key, "stepsize") == 0) {
      stepsize_init = atof(co->value);
    } else if (strcmp(co->key, "optim_maxiter") == 0) {
      maxoptimiter = atoi(co->value);
    } else if (strcmp(co->key, "gtol") == 0) {
      gtol = atof(co->value);
    } else if (strcmp(co->key, "ls_maxiter") == 0) {
      ls_maxiter = atoi(co->value);
    } else if (strcmp(co->key, "ls_factor") == 0) {
      ls_factor = atof(co->value);
    } else if (strcmp(co->key, "weights_open_init") == 0) {
      weights_open_init = atof(co->value);
    } else if (strcmp(co->key, "type_openlayer") == 0) {
      if (strcmp(co->value, "replicate") == 0) {
        openlayer_type = 0;
      } else if (strcmp(co->value, "activate") == 0) {
        openlayer_type = 1;
      } else {
        printf("Invalid type_openlayer!\n");
        MPI_Finalize();
        return (0);
      }
    } else if (strcmp(co->key, "weights_init") == 0) {
      weights_init = atof(co->value);
    } else if (strcmp(co->key, "weights_class_init") == 0) {
      weights_class_init = atof(co->value);
    } else if (strcmp(co->key, "hessian_approx") == 0) {
      if (strcmp(co->value, "BFGS") == 0) {
        hessianapprox_type = BFGS_SERIAL;
      } else if (strcmp(co->value, "L-BFGS") == 0) {
        hessianapprox_type = LBFGS;
      } else if (strcmp(co->value, "Identity") == 0) {
        hessianapprox_type = IDENTITY;
      } else {
        printf("Invalid Hessian approximation!");
        return -1;
      }
    } else if (strcmp(co->key, "lbfgs_stages") == 0) {
      lbfgs_stages = atoi(co->value);
    } else if (strcmp(co->key, "validationlevel") == 0) {
      validationlevel = atoi(co->value);
    }
    if (co->prev != NULL) {
      co = co->prev;
    } else {
      break;
    }
  }

  /* Sanity checks */
  if (nfeatures > nchannels || nclasses > nchannels) {
    printf("ERROR! Choose a wider netword!\n");
    printf(" -- nFeatures = %d\n", nfeatures);
    printf(" -- nChannels = %d\n", nchannels);
    printf(" -- nClasses = %d\n", nclasses);
    exit(1);
  }

  if (NI_num_tols != NI_levels) {
    printf("ERROR! Number of entered NI tolerances (NI_tols) must equal number of NI levels (NI_levels)!\n"); 
    printf(" -- num NI_tols = %d\n", NI_num_tols);
    printf(" -- NI_levels = %d\n", NI_levels); 
    exit(1);
  }

  if ( (NI_interp_type != 0) && (NI_interp_type != 1)) {
    printf("ERROR! NI_interp_type incorrect, only 0 (piece-wise constant) and 1 (linear) supported!\n"); 
    printf(" -- NI_interp_type = %d\n", NI_interp_type);
    exit(1);
  }

  return 0;
}

Config::config_option *Config::parsefile(char *path) {
  FILE *fp;

  if ((fp = fopen(path, "r+")) == NULL) {
    perror("fopen()");
    return NULL;
  }

  config_option *last_co_addr = NULL;

  while (1) {
    config_option *co = NULL;
    if ((co = (config_option *)calloc(1, sizeof(config_option))) == NULL)
      continue;
    memset(co, 0, sizeof(struct config_option));
    co->prev = last_co_addr;

    if (fscanf(fp, "%s = %s", &co->key[0], &co->value[0]) != 2) {
      if (feof(fp)) {
        break;
      }
      if (co->key[0] == '#') {
        while (fgetc(fp) != '\n') {
          // Do nothing (to move the cursor to the end of the line).
        }
        free(co);
        continue;
      }
      perror("fscanf()");
      free(co);
      continue;
    }
    // printf("Key: %s\nValue: %s\n", co->key, co->value);
    last_co_addr = co;
  }
  return last_co_addr;
}

int Config::writeToFile(FILE *outfile) {
  const char *activname, *networktypename, *hessetypename, *optimtypename,
      *stepsizetypename;

  /* Get names of some int options */
  switch (activation) {
    case TANH:
      activname = "tanh";
      break;
    case RELU:
      activname = "ReLu";
      break;
    case SMRELU:
      activname = "SmoothReLU";
      break;
    default:
      activname = "invalid!";
  }
  switch (network_type) {
    case DENSE:
      networktypename = "dense";
      break;
    case CONVOLUTIONAL:
      networktypename = "convolutional";
      break;
    default:
      networktypename = "invalid!";
  }
  switch (hessianapprox_type) {
    case BFGS_SERIAL:
      hessetypename = "BFGS";
      break;
    case LBFGS:
      hessetypename = "L-BFGS";
      break;
    case IDENTITY:
      hessetypename = "Identity";
      break;
    default:
      hessetypename = "invalid!";
  }
  switch (batch_type) {
    case DETERMINISTIC:
      optimtypename = "deterministic";
      break;
    case STOCHASTIC:
      optimtypename = "stochastic";
      break;
    default:
      optimtypename = "invalid!";
  }
  switch (stepsize_type) {
    case FIXED:
      stepsizetypename = "fixed";
      break;
    case BACKTRACKINGLS:
      stepsizetypename = "backtracking line-search";
      break;
    case ONEOVERK:
      stepsizetypename = "1/k";
      break;
    default:
      stepsizetypename = "invalid!";
  }

  /* print config option */
  fprintf(outfile, "# Problem setup: datafolder           %s \n", datafolder);
  fprintf(outfile, "#                training examples    %s \n", ftrain_ex);
  fprintf(outfile, "#                training labels      %s \n",
          ftrain_labels);
  fprintf(outfile, "#                validation examples  %s \n", fval_ex);
  fprintf(outfile, "#                validation labels    %s \n", fval_labels);
  fprintf(outfile, "#                ntraining            %d \n", ntraining);
  fprintf(outfile, "#                nvalidation          %d \n", nvalidation);
  fprintf(outfile, "#                nfeatures            %d \n", nfeatures);
  fprintf(outfile, "#                nclasses             %d \n", nclasses);
  fprintf(outfile, "#                nchannels            %d \n", nchannels);
  fprintf(outfile, "#                nlayers              %d \n", nlayers);
  fprintf(outfile, "#                T                    %f \n", T);
  fprintf(outfile, "#                network type         %s \n",
          networktypename);
  fprintf(outfile, "#                Activation           %s \n", activname);
  fprintf(outfile, "#                openlayer type       %d \n",
          openlayer_type);
  fprintf(outfile, "# N Iter setup:  NI levels            %d \n",
          NI_levels);
  fprintf(outfile, "#                NI rfactor           %d \n",
          NI_rfactor);
  fprintf(outfile, "#                NI interp type       %d \n",
          NI_interp_type);
  for(int i = 0; i < NI_levels; i++){
     fprintf(outfile, "#                NI tols[%d]          %d \n",
                i, NI_tols[i]);
  }
  fprintf(outfile, "# XBraid setup:  max levels           %d \n",
          braid_maxlevels);
  fprintf(outfile, "#                min coarse           %d \n",
          braid_mincoarse);
  fprintf(outfile, "#                coasening            %d \n",
          braid_cfactor);
  fprintf(outfile, "#                coasening (level 0)  %d \n",
          braid_cfactor0);
  fprintf(outfile, "#                max. braid iter      %d \n",
          braid_maxiter);
  fprintf(outfile, "#                abs. tol             %1.e \n",
          braid_abstol);
  fprintf(outfile, "#                abs. toladj          %1.e \n",
          braid_abstoladj);
  fprintf(outfile, "#                print level          %d \n",
          braid_printlevel);
  fprintf(outfile, "#                access level         %d \n",
          braid_accesslevel);
  fprintf(outfile, "#                skip?                %d \n",
          braid_setskip);
  fprintf(outfile, "#                fmg?                 %d \n", braid_fmg);
  fprintf(outfile, "#                nrelax (level 0)     %d \n",
          braid_nrelax0);
  fprintf(outfile, "#                nrelax               %d \n", braid_nrelax);
  fprintf(outfile, "# Optimization:  optimization type    %s \n",
          optimtypename);
  fprintf(outfile, "#                nbatch               %d \n", nbatch);
  fprintf(outfile, "#                gamma_tik            %1.e \n", gamma_tik);
  fprintf(outfile, "#                gamma_ddt            %1.e \n", gamma_ddt);
  fprintf(outfile, "#                gamma_class          %1.e \n",
          gamma_class);
  fprintf(outfile, "#                stepsize type        %s \n",
          stepsizetypename);
  fprintf(outfile, "#                stepsize             %f \n",
          stepsize_init);
  fprintf(outfile, "#                max. optim iter      %d \n", maxoptimiter);
  fprintf(outfile, "#                gtol                 %1.e \n", gtol);
  fprintf(outfile, "#                max. ls iter         %d \n", ls_maxiter);
  fprintf(outfile, "#                ls factor            %f \n", ls_factor);
  fprintf(outfile, "#                weights_init         %f \n", weights_init);
  fprintf(outfile, "#                weights_open_init    %f \n",
          weights_open_init);
  fprintf(outfile, "#                weights_class_init   %f \n",
          weights_class_init);
  fprintf(outfile, "#                hessianapprox_type   %s \n",
          hessetypename);
  fprintf(outfile, "#                lbfgs_stages         %d \n", lbfgs_stages);
  fprintf(outfile, "#                validationlevel      %d \n",
          validationlevel);
  fprintf(outfile, "\n");

  return 0;
}

MyReal Config::getStepsize(int optimiter) {
  MyReal stepsize = 0.0;

  switch (stepsize_type) {
    case FIXED:
      stepsize = stepsize_init;
      break;
    case BACKTRACKINGLS:
      stepsize = stepsize_init;
      break;
    case ONEOVERK:
      stepsize = 1.0 / (MyReal)(optimiter +
                                1);  // add one because optimiter starts with 0
  }

  return stepsize;
}

void Config::string_to_intarray(char* str, int** NI_tols, int* NI_num_tols)
{
   /* We hardcode in a length limit of 100 for the possible number of integers */
   int   len = 0;
   int   temp[100];
   char* ptr;

   while( *str != '\0' ){
      
      if (isdigit(*str)){
         /* If we end up wanting floats, instead ints, you'll need to  use 
          * "strtof(str, &ptr)", and change some types, i.e., NI_tols to float* */
         temp[len++] = strtol(str, &ptr, 10);
      }
      else {
         ptr = str + 1;
      }

      str = ptr;
   }
   
   /* Copy over output */
   (*NI_tols) = (int*) malloc(len*sizeof(int));
   for(int i = 0; i < len; i++){
      (*NI_tols)[i] = temp[i];
   }
   *NI_num_tols = len;

}
