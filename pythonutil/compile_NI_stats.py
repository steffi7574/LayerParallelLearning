import scipy as sp
from matplotlib import pyplot as plt
import os
import sys
import copy

def find_val_acc(lines, num_occurences):
    '''
    take in list of file lines, return number after num occurrences of
      "Final validation accuracy:  78.50%"
    Otherwise, return nan
    '''
    seen = 0
    for line in lines:
        if line[:26] == "Final validation accuracy:":
            seen += 1
            val_acc = float(line[27:line.rfind("%")])
    if seen == num_occurences: return val_acc
    else: return sp.nan

def print_stats(vals):
    '''
    Helper function to print stats
    '''
    indys = (vals == 0) + sp.isnan(vals) 
    indys2 = sp.setdiff1d(sp.arange(len(vals)), indys.nonzero()[0]) 
    vals = vals[indys2]
    print("  Mean   : " + str(sp.mean(vals) ) + " %")
    print("  Median : " + str(sp.median(vals) ) + " %")
    print("  Max    : " + str(max(vals) ) + " %")
    print("  Min    : " + str(min(vals) ) + " %")
    print("  Std Dev: " + str(sp.std(vals) ) + " %")
    print("  Nan    : " + str(len(indys.nonzero()[0])))

    return indys2

##
# Directories to compare, choose two directories to be directory and directory_reference
##
#
# Peaks with nested iter
directory = "peaks_NI_experiments"
#
# Peaks with no nested iteration, but 50% more optimization iters to make this
# algorithm cost roughly as much as nested iteration
directory_reference = "peaks_experiments_longer"
#
# Same as peaks_experiments, but with the same number of optimization iters as nested iteration 
#directory_reference = "peaks_experiments"


sys.path.append('./'+directory+'/')
from config import *
configfile = Config(directory+"/peaks.cfg")

val_accuracies = []
val_accuracies_ref = []
val_accuracies_diff = []
job_names = []

################################
# Copy this section from submit_parameterstudy.py

# Specify the varying parameters
nlayers    = [24]
#
gammatik   = [1e-7]
gammaddt   = [1e-5]
gammaclass = [1e-7, 1e-2]
#
weights_open_init  = [1e-3, 1e-1]
weights_init       = [0,    1e-6]
weights_class_init = [1e-3, 1e-1]
#
NI_levels      = [2]
NI_rfactor     = [2]
NI_interp_type = [0, 1]
NI_tols        = ["75,75", "90,90"]

# number of processors
# want that at least nlayers / cf
npt = 12 

# Submit a job for each parameter setup
for nl in range(len(nlayers)):

    for itik in range(len(gammatik)):

        for iddt in range(len(gammaddt)):

            for iclass in range(len(gammaclass)):

                for wopen in range(len(weights_open_init)):
                    
                    for winit in range(len(weights_init)):
                    
                        for wclass in range(len(weights_class_init)):
                        
                            for nilevels in range(len(NI_levels)):
                            
                                for nirfactor in range(len(NI_rfactor)):
                                
                                    for niinterp in range(len(NI_interp_type)):
                                    
                                        for nitols in range(len(NI_tols)):

                                            # Copy the global config file
                                            konfig = copy.deepcopy(configfile)
                                            
                                            # Change the config entry 
                                            konfig.nlayers     = nlayers[nl]
                                            #
                                            konfig.gamma_tik   = gammatik[itik]
                                            konfig.gamma_ddt   = gammaddt[iddt]
                                            konfig.gamma_class = gammaclass[iclass]
                                            #
                                            konfig.weights_open_init = weights_open_init[wopen]
                                            konfig.weights_init = weights_init[winit]
                                            konfig.weights_class_init = weights_class_init[wclass]
                                            #
                                            konfig.NI_levels = NI_levels[nilevels] 
                                            konfig.NI_rfactor = NI_rfactor[nirfactor] 
                                            konfig.NI_interp_type = NI_interp_type[niinterp] 
                                            konfig.NI_tols = NI_tols[nitols] 

                                            # Specify jobname 
                                            jobname =  \
                                                      "peaksOpt"  +\
                                                      "_nl"      + str(konfig.nlayers)   +\
                                                      "_tik"     + str(konfig.gamma_tik)   +\
                                                      "_ddt"     + str(konfig.gamma_ddt)   +\
                                                      "_class"   + str(konfig.gamma_class) +\
                                                      "_wo"      + str(konfig.weights_open_init ) +\
                                                      "_wi"      + str(konfig.weights_init ) +\
                                                      "_wc"      + str(konfig.weights_class_init ) +\
                                                      "_nilvl"   + str(konfig.NI_levels ) +\
                                                      "_nirf"    + str(konfig.NI_rfactor ) +\
                                                      "_niinterp"+ str(konfig.NI_interp_type ) +\
                                                      "_nitols"  + str(konfig.NI_tols )
                                            
                                            # End Copy this section from submit_parameterstudy.py
                                            ################################
                                            
                                            # Change above jobname here, (1) remove ni params, and 
                                            # (2) change nlayers to be the final nlayers
                                            jobname_reference =  \
                                                      "peaksOpt"  +\
                                                      "_nl"      + str( (konfig.nlayers-2)*( 2**(konfig.NI_levels-1)) + 2 )   +\
                                                      "_tik"     + str(konfig.gamma_tik)   +\
                                                      "_ddt"     + str(konfig.gamma_ddt)   +\
                                                      "_class"   + str(konfig.gamma_class) +\
                                                      "_wo"      + str(konfig.weights_open_init ) +\
                                                      "_wi"      + str(konfig.weights_init ) +\
                                                      "_wc"      + str(konfig.weights_class_init )

                                            ##
                                            # Load two output files, and find val accuracy
                                            try:
                                                f = open(directory + '/' + jobname + "/run.out")
                                                lines = f.readlines()
                                                val_accuracies.append(find_val_acc(lines, konfig.NI_levels))
                                                f.close()
                                                if sp.isnan(val_accuracies[-1]) or val_accuracies[-1] == 0:
                                                    print("Zero or nan val accuracy,  " + str(val_accuracies[-1]) + ",  " + jobname)
                                            except:
                                                print("\nCan't find " + jobname + "\n")
                                                val_accuracies.append(sp.nan) 
                                            
                                            try:
                                                f = open(directory_reference + '/' + jobname_reference + "/run.out")
                                                lines = f.readlines()
                                                val_accuracies_ref.append(find_val_acc(lines, 1))
                                                f.close()
                                            except:
                                                print("\nCan't find REFERENCE " + jobname_reference + "\n")
                                                val_accuracies_ref.append(sp.nan) 

                                            val_accuracies_diff.append( val_accuracies[-1] - val_accuracies_ref[-1])
                                            job_names.append(jobname)
                                            # Account for missed filenames ... 

# Need to account for entries that are "nan" and "0"
val_accuracies = sp.array(val_accuracies) 
val_accuracies_diff = sp.array(val_accuracies_diff)
val_accuracies_ref = sp.array(val_accuracies_ref)


## Print Mean, median, min, max, st dev of validation accuracies
print("\n------------------------------------------")
print("Nested Iteration Validation Accuracy Statistics")
indys2 = print_stats(val_accuracies)

print("\nNormal, Non-Nested Iteration Validation Accuracy Statistics")
print_stats(val_accuracies_ref)

print("\nStats for Nested Iteration MINUS Normal Experiments")
print_stats(val_accuracies_diff[indys2])

## Print top five NI results
indys3 = sp.argsort(val_accuracies[indys2])[-5:]
print("\n------------------------------------------")
print("\nTop 5 Nested Iteration Validation accuracies")
print( [ "%1.2f %%"%v for v in (val_accuracies[indys2])[indys3] ] )
print("\nCorresponding Non-Nested Iteration Validation accuracies")
print( [ "%1.2f %%"%v for v in (val_accuracies_ref[indys2])[indys3] ] )
print("\nExperiments located in:")
print((sp.asarray(job_names)[indys2])[indys3])

## Print bottom five NI results
indys4 = sp.argsort(val_accuracies[indys2])[0:5]
print("\n------------------------------------------")
print("\nWorst 5 Nested Iteration Validation accuracies")
print( [ "%1.2f %%"%v for v in (val_accuracies[indys2])[indys4] ] )
print("\nCorresponding Non-Nested Iteration Validation accuracies")
print( [ "%1.2f %%"%v for v in (val_accuracies_ref[indys2])[indys4] ] )
print("\nExperiments located in:")
print((sp.asarray(job_names)[indys2])[indys4])


