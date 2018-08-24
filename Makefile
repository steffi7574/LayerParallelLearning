CC     =gcc
CXX    =g++
MPICC  = mpicc
MPICXX = mpicxx

INC = -I. -I$(BRAID_INC_DIR) -I$(CODI_DIR)
BRAID_INC_DIR = /home/sguenther/Software/xbraid/braid
BRAID_LIB_FILE = /home/sguenther/Software/xbraid/braid/libbraid.a
CODI_DIR = /home/sguenther/Software/CoDiPack_v1.0/include/

# set compiler flags
CPPFLAGS= -g -Wall -pedantic -lm -Wno-write-strings -Wno-delete-non-virtual-dtor

DEPS = lib.hpp braid_wrapper.hpp hessianApprox.hpp parser.h layer.hpp linalg.hpp network.hpp
OBJ-pint   = main.o lib.o braid_wrapper.o hessianApprox.o Layer.o linalg.o
OBJ-awesome = main-awesome.o lib.o hessianApprox.o layer.o linalg.o network.o

%.o: %.cpp $(DEPS)
	$(MPICXX) $(CPPFLAGS) -c $< -o $@  $(INC)

main-awesome: $(OBJ-awesome)
	$(MPICXX) $(CPPFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

main: $(OBJ-pint)
	$(MPICXX) $(CPPFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

clean: 
	rm -f *.o
	rm -f main
