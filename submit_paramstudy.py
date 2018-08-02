#!/usr/bin/eny python
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
ntraining = [10]

# Submit a job for each parameter setup
for i in range(len(ntraining)):

    # Copy the global config file
    konfig = copy.deepcopy(configfile)
    
    # Change the config entry 
    konfig.ntraining = ntraining[i]

    # Specify jobname 
    jobname = "ntrain" + str(konfig.ntraining) 
    
    # create folder for the job
    if os.path.exists(jobname):
       pass 
    else:
       os.mkdir(jobname)

    # create a link to training and validation data
    make_link(datafolder,jobname + "/" + datafolder)
    #os.symlink("/home/sguenther/Numerics/DNN_PinT/"+datafolder, datafolder)
    
    # Create a config file
    newconfigfile = jobname + ".cfg"
    konfig.dump(jobname + "/" + newconfigfile)

    # submit the job
    os.chdir(jobname)
    submit_job(jobname, runcommand, 1, "01:00:00","../main", newconfigfile)
    os.chdir("../")
