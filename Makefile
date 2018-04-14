CC=gcc
MPICC = mpicc

INC = -I. -I$(BRAID_INC_DIR)
BRAID_INC_DIR = /home/sguenther/Software/braid_llnl/
BRAID_LIB_FILE = /home/sguenther/Software/braid_llnl/libbraid.a

# set compiler flags
CFLAGS= -g -Wall -pedantic -lm

DEPS = lib.h
OBJ-serial = main-serial.o lib.o
OBJ-pint   = main.o lib.o

%.o: %.c $(DEPS)
	$(MPICC) $(CFLAGS) -c $< -o $@  $(INC)

main-serial: $(OBJ-serial)
	$(MPICC) $(CFLAGS) -o $@ $^ 

main: $(OBJ-pint)
	$(MPICC) $(CFLAGS) -o $@ $^ $(BRAID_LIB_FILE)

clean: 
	rm *.o
	rm main
