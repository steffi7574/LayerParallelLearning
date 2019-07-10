# Layer-parallel training of deep residual neural networks 
This code performs layer-parallel training of deep neural networks of residual type. It utilizes the parallel-in-time software library [XBraid](https://github.com/XBraid/xbraid) to distribute layers of the network to different compute units. Instead of sequential forward and backward propagation through the network, iterative multigrid udpates are performed in parallel to solve for the network propagation and the training simultaneously. See the paper [Guenther et al.](https://arxiv.org/pdf/1812.04352.pdf) for a describtion of the method and all details.

## Build
The repository includes XBraid as a submodule. To clone both, use either `git clone --recurse-submodules [...]` for Git version >= 2.13, or `git clone [...]` followed by `cd xbraid`, `git submodule init` and `git submodule update` for older Git versions. 

Type `make` in the main directory to build both the code and the XBraid library. 

## Run
Test cases are located in the 'examples/' subfolder. Each example contains a `*.cfg` that holds configuration options for the current example dataset, the layer-parallelization with XBraid, and the optimization method and parameters. 

Run the test cases by callying './main' with the corresponding configuration file, e.g. `./main examples/peaks/peaks.cfg`

## Output
An optimization history file 'optim.dat' will be flushed to the examples subfolder. 

## Roland / Stef combo - LEARNING/TODO:

### Overview:
* Xbraid => 3 braid apps know one net
* One NN
* Braid has to be initialized first -> Decides about layers per core
* Tangling with NN is given due to reason above
    - create nn
    - give nn to braid
    - init nn

## 2Dos 😊
* [X] Clean up syntax (google std)
* [ ] Associate code with [paper](https://arxiv.org/pdf/1812.04352.pdf)
* [ ] Refactoring (data(ste)) = Make code more readable by applying [SOLID](https://en.wikipedia.org/wiki/SOLID) principles
  - [ ] Single Responsibility
  - [ ] Open-Closed
  - [ ] Liskov Substitution
  - [ ] Interface Segregation
  - [ ] Dependency Inversion
* [ ] Extract methods and clean up objects
* [ ] Comment everything
