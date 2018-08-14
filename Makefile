CC     =gcc
CXX    =g++
MPICC  = mpicc
MPICXX = mpicxx

INC = -I. -I$(BRAID_INC_DIR) -I$(CODI_DIR)
BRAID_INC_DIR = /home/sguenther/Software/xbraid/braid
BRAID_LIB_FILE = /home/sguenther/Software/xbraid/braid/libbraid.a
CODI_DIR = /home/sguenther/Software/CoDiPack_v1.0/include/

# set compiler flags
CFLAGS= -g -Wall -pedantic -lm -Wno-write-strings

DEPS = lib.h braid_wrapper.h bfgs.h l-bfgs.hpp
OBJ-pint   = main.o lib.o bfgs.o braid_wrapper.o l-bfgs.o

%.o: %.c $(DEPS)
	$(MPICXX) $(CFLAGS) -c $< -o $@  $(INC)

main: $(OBJ-pint)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

main-prop: $(OBJ-val)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

clean: 
	rm -f *.o
	rm -f main
	rm -f main-prop
