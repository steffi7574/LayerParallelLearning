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

DEPS = lib.hpp braid_wrapper.hpp HessianApprox.hpp parser.h
OBJ-pint   = main.o lib.o braid_wrapper.o HessianApprox.o

%.o: %.cpp $(DEPS)
	$(MPICXX) $(CPPFLAGS) -c $< -o $@  $(INC)

main: $(OBJ-pint)
	$(MPICXX) $(CPPFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

clean: 
	rm -f *.o
	rm -f main
