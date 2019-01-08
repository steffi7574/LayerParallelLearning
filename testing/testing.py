#!/usr/bin/env python

import sys 
import os
import copy
import subprocess
import string
from config import *
from util import *


# Specify testcase
case = "peaks"

# Get the global config file
config = Config(case + ".cfg")

# Specify the configuration to be tested
braid_maxlevelslist = [1,10]

# Specify number of processors to be tested
nptlist = [1,2,5]

# Iterate over configuration
for j,ml in enumerate(braid_maxlevelslist):

    # Iterate over number of processors
    for i,npt in enumerate(nptlist):
    
        # Create testing folder
        testname = case + "test_ml" + str(ml) + "_npt" + str(npt)
        if os.path.exists(testname):
           pass
        else:
           os.mkdir(testname)
       
        # create a link to training and validation data
        datafolder = "../" + config.datafolder
        make_link(datafolder, testname + "/" + config.datafolder)
        
        # Set the new configuration
        konfig = copy.deepcopy(config)
        konfig.braid_maxlevels = ml 
    
        # create the config file
        testconfig = testname + ".cfg"
        konfig.dump(testname + "/" + testconfig)
        
        # run the test
        os.chdir(testname)
        runcommand = "mpirun -n " + str(npt) + " ../../main " + testconfig + " > tmp"
        print("Running Test: " + testname)
        #print("  " + runcommand)
        subprocess.call(runcommand, shell=True)
        os.chdir("../")
        
        # compare output file to the reference
        refname = case + ".ml" + str(ml) + ".optim.ref"
        err = comparefiles(refname, testname + "/optim.dat")
        
        # Print result
        if (err > 0):
            print("  !!! Test failed !!!")
            print("  vimdiff " + refname + " " + testname + "/optim.dat")
        else:
            print("  Test passed!")
        
