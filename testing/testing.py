#!/usr/bin/env python

import sys
import argparse
import os
import copy
import subprocess
import string
from config import *
from util import *

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--case', help='name of test case', default='peaks')
parser.add_argument('-ml', '--maxlevels', type=int, nargs='+', help='braid_maxlevels to be tested',  default=[1,10])
parser.add_argument('-npt', '--nprocs', type=int, nargs='+', help='number of processors to be tested',  default=[1,2,5])

# Parse command line arguments
args = parser.parse_args()
case = args.case
nptlist  = args.nprocs
braid_maxlevelslist = args.maxlevels
print("Testing case \"" + case +  "\", npt=" + str(nptlist) + ", braid_maxlevels=" + str(braid_maxlevelslist))

# Specify the output file to compare
outfile = "optim.dat"

# Get the global config file
config = Config(case + ".cfg")

# Iterate over configuration
for j,ml in enumerate(braid_maxlevelslist):

    # Iterate over number of processors
    for i,npt in enumerate(nptlist):

        # Set the test case name 
        testname = case + ".npt" + str(npt) + ".ml" + str(ml) 
    
        # Create testing folder
        testfoldername = "test." + testname
        if os.path.exists(testfoldername):
           pass
        else:
           os.mkdir(testfoldername)
       
        # create a link to training and validation data
        datafolder = "../" + config.datafolder
        make_link(datafolder, testfoldername + "/" + config.datafolder)
        
        # Set the new configuration
        konfig = copy.deepcopy(config)
        konfig.braid_maxlevels = ml 
    
        # create the config file
        testconfig = testname + ".cfg"
        konfig.dump(testfoldername + "/" + testconfig)
        
        # run the test
        os.chdir(testfoldername)
        runcommand = "mpirun -n " + str(npt) + " ../../main " + testconfig + " > tmp"
        print("Running Test: " + testname)
        #print("  " + runcommand)
        subprocess.call(runcommand, shell=True)
        os.chdir("../")
        
        # compare output file to the reference
        refname = testname  + "." + outfile 
        err = comparefiles(refname, testfoldername + "/" + outfile)
        
        # Print result
        if (err > 0):
            print("  !!! Test failed !!!")
            print("  vimdiff " + refname + " " + testfoldername + "/" + outfile)
        else:
            print("  Test passed!")
