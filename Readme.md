# Layer-parallel training of deep residual neural networks 

This code performs layer-parallel training of deep neural networks of residual type. See the paper [some].


## Installation

Clone the repository including the XBraid submodule with
    git clone --recurse-submodules https://github.com/steffi7574/DNN_PinT.git
Type
    make
to build the XBraid library and the main code. 

## Running

Test cases are located in the 'examples/' subfolder. Each example contains a '\*.cfg' that holds configuration options for the current example dataset, the layer-parallelization with XBraid, and the optimization method and parameters. 

Run the test cases by callying './main' with the corresponding configuration file, e.g.  
    ./main examples/peaks/peaks.cfg

## Output
An optimization history file 'optim.dat' will be flushed to the examples subfolder. 

