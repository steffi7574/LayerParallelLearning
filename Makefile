CC     = mpicc
CXX    = mpicxx

INC = -I. -I$(BRAID_INC_DIR) -I$(OPTIM_INC_DIR)
BRAID_INC_DIR = xbraid/braid
BRAID_LIB_FILE = xbraid/braid/libbraid.a
OPTIM_INC_DIR = optimlib
OPTIM_LIB_FILE = optimlib/liboptim.a

# set compiler flags
CPPFLAGS = -g -Wall -pedantic -lm -Wno-write-strings -Wno-delete-non-virtual-dtor -std=c++11

DEPS = braid_wrapper.hpp layer.hpp network.hpp util.hpp config.hpp dataset.hpp opt_wrapper.hpp
OBJ= main.o util.o layer.o network.o braid_wrapper.o config.o dataset.o opt_wrapper.o

.PHONY: all $(BRAID_LIB_FILE) $(OPTIM_LIB_FILE) clean

all: main $(BRAID_LIB_FILE) $(OPTIM_LIB_FILE)

%.o: %.cpp $(DEPS)
	$(CXX) $(CPPFLAGS) -c $< -o $@  $(INC)

$(BRAID_LIB_FILE):
	cd xbraid; make braid

$(OPTIM_LIB_FILE):
	cd $(OPTIM_INC_DIR); make

main:  $(OBJ) $(BRAID_LIB_FILE) $(OPTIM_LIB_FILE)
	$(CXX) $(CPPFLAGS) -o $@ $^

clean: 
	rm -f *.o
	rm -f main
