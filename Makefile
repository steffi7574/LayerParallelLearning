CC     = mpicc
CXX    = mpicxx

INC = -I. -I$(BRAID_INC_DIR) -I$(CODI_DIR)
BRAID_INC_DIR = /home/sguenther/Software/xbraid/braid
BRAID_LIB_FILE = /home/sguenther/Software/xbraid/braid/libbraid.a
CODI_DIR = /home/sguenther/Software/CoDiPack_v1.0/include/

# set compiler flags
CPPFLAGS = -g -Wall -pedantic -lm -Wno-write-strings -Wno-delete-non-virtual-dtor -std=c++11

DEPS = lib.hpp braid_wrapper.hpp hessianApprox.hpp parser.h layer.hpp linalg.hpp network.hpp util.hpp
OBJ-pint   = main.o lib.o braid_wrapper.o hessianApprox.o layer.o linalg.o
OBJ-awesome = main-awesome.o util.o hessianApprox.o layer.o linalg.o network.o braid_wrapper.o

%.o: %.cpp $(DEPS)
	$(CXX) $(CPPFLAGS) -c $< -o $@  $(INC)

main-awesome: $(OBJ-awesome)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

clean: 
	rm -f *.o
	rm -f main
	rm -f main-awesome
