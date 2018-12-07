CC     = mpicc
CXX    = mpicxx

INC = -I. -I$(BRAID_INC_DIR)
BRAID_INC_DIR = /home/eccyr/Projects/ECRP/Code/convnet/xbraid/braid
BRAID_LIB_FILE = /home/eccyr/Projects/ECRP/Code/convnet/xbraid/braid/libbraid.a

# set compiler flags
CPPFLAGS = -O -pg -g -Wall -pedantic -lm -Wno-write-strings -Wno-delete-non-virtual-dtor -std=c++11

DEPS = braid_wrapper.hpp hessianApprox.hpp parser.h layer.hpp linalg.hpp network.hpp util.hpp
OBJ= main.o util.o hessianApprox.o layer.o linalg.o network.o braid_wrapper.o

%.o: %.cpp $(DEPS)
	$(CXX) $(CPPFLAGS) -c $< -o $@  $(INC)

main: $(OBJ)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

clean: 
	rm -f *.o
	rm -f main
