CC     = mpicc
CXX    = mpicxx

INC = -I. -I$(BRAID_INC_DIR)
BRAID_INC_DIR = xbraid/braid
BRAID_LIB_FILE = xbraid/braid/libbraid.a

# set compiler flags
CPPFLAGS = -g -Wall -pedantic -lm -Wno-write-strings -Wno-delete-non-virtual-dtor -std=c++11

DEPS = braid_wrapper.hpp hessianApprox.hpp layer.hpp linalg.hpp network.hpp util.hpp config.hpp dataset.hpp bsplines.hpp
OBJ= main.o util.o hessianApprox.o layer.o linalg.o network.o braid_wrapper.o config.o dataset.o bsplines.o

.PHONY: all $(BRAID_LIB_FILE) clean

all: main $(BRAID_LIB_FILE)

%.o: %.cpp $(DEPS)
	$(CXX) $(CPPFLAGS) -c $< -o $@  $(INC)

$(BRAID_LIB_FILE):
	cd xbraid; make braid

main: $(BRAID_LIB_FILE) $(OBJ) 
	$(CXX) $(CPPFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

clean: 
	rm -f *.o
	rm -f main
