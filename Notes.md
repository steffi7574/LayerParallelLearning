TODO 

- Ask Lars about this simple convolutional network architecture.  Is it relevant (enough)?

- Implement ApplyFWD

- Implement ApplyBWD
   
- Add options to config.cfg that allow for the choice of a convolutional
  network, and the setting of the number of convolutions and the stencil size
  of the convolutions.  This will impact the parsing of config.cfg in main.cpp  

  Then, in network.cpp, new if-statements will be needed to create convolutional layers, 
  not dense layers

- Likely sizing issues, such as the nchannels*nchannels assumption in 
   network.cpp::evalRegulDDT()

   Probably other sizing issues


- Dump MNIST data into a input files

