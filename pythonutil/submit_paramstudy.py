#!/usr/bin/env python

import sys 
import os
import copy
from batch_job import submit_job
from config import *
from util import *

# Specify runcommand for the cluster ("srun -n" on quartz or "mpirun -np" on elwe)
runcommand = "srun -n"

# Specify name of training data folder
datafolder = "data"

# Specify the global config file 
configfile = Config("config.cfg")

# Specify the varying parameters
gammatik   = [1e-5]
gammaddt   = [1e-5]
gammaclass = [1e-1, 1e-5]
npt = 36

# Submit a job for each parameter setup
for itik in range(len(gammatik)):

    for iddt in range(len(gammaddt)):

        for iclass in range(len(gammaclass)):

            # Copy the global config file
            konfig = copy.deepcopy(configfile)

            # Change the config entry 
            konfig.gamma_tik   = gammatik[itik]
            konfig.gamma_ddt   = gammaddt[iddt]
            konfig.gamma_class = gammaclass[iclass]

            # Specify jobname 
            jobname =  \
                      "n1024opt"  +\
                      "tik"     + str(konfig.gamma_tik)   +\
                      "ddt"     + str(konfig.gamma_ddt)   +\
                      "class"   + str(konfig.gamma_class)

            # create folder for the job
            if os.path.exists(jobname):
               pass
            else:
               os.mkdir(jobname)

            # create a link to training and validation data
            make_link(datafolder,jobname + "/" + datafolder)

            # Create a config file
            newconfigfile = jobname + ".cfg"
            konfig.dump(jobname + "/" + newconfigfile)

            # submit the job
            os.chdir(jobname)
            submit_job(jobname, runcommand, npt, "10:00:00", "../main", newconfigfile)
            os.chdir("../")

