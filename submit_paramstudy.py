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
ntraining = [10,20]

# prepare gnuplot script 
plotfile = open("plot_optim.plt", "w")
plotfile.write("reset\n")
plotfile.write("set y2label 'accuracy'\n")
plotfile.write("set ylabel 'objective'\n")
plotfile.write("set yrange[0:1.8]\n")
plotfile.write("set key outside\n")
plotfile.write("set key left\n")
plotfile.write("set y2tics\n")
plotfile.write("set ytics nomirror\n")
plotfile.write("\nplot \\ \n")
plotlc = 1
# prepare key file for gnuplot
keyfile = open("plot_keys.plt","w")
keyfile.write("reset\n")
keyfile.write("set yrange[0:0.1]\n")
keyfile.write("set xrange[0:0.1]\n")
keyfile.write("set key outside\n")
keyfile.write("\nplot \\ \n")

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
    #submit_job(jobname, runcommand, 1, "01:00:00","../main", newconfigfile)
    os.chdir("../")

    # add to plotscript 
    plotstring  = "'" + jobname + "/optim.dat' u 1:4  axis x1y1 w l lc " + str(plotlc) + " dt 1 notitle, \\ \n"
    plotstring += "'" + jobname + "/optim.dat' u 1:11 axis x1y2 w l lc " + str(plotlc) + " dt 1 notitle, \\ \n"
    plotstring += "'" + jobname + "/optim.dat' u 1:12 axis x1y2 w l lc " + str(plotlc) + " dt 2 notitle, \\ \n"
    #plotstring += "sin(x) - 100 lc " + str(plotlc) + " title '" + jobname + "', \\ \n"
    keystring = "sin(x) - 100 lc " + str(plotlc) + " title '" + jobname + "', \\ \n"
    plotfile.write(plotstring) 
    keyfile.write(keystring) 
    plotlc = plotlc + 1


# close gnuplot file
plotfile.close()
keyfile.close()
