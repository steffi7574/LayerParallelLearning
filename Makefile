SRC_DIR   = src
INC_DIR   = include
BUILD_DIR = build

#list all source files in SRC_DIR
SRC_FILES  = $(wildcard $(SRC_DIR)/*.cpp)
SRC_FILES += $(wildcard $(SRC_DIR)/*/*.cpp)
OBJ_FILES  = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))

# XBraid location
BRAID_INC_DIR = xbraid/braid
BRAID_LIB_FILE = xbraid/braid/libbraid.a

# set inc dir
INC = -I$(INC_DIR) -I$(BRAID_INC_DIR)

# set compiler flags
CXX_FLAGS = -g -Wall -pedantic -lm -Wno-write-strings -Wno-delete-non-virtual-dtor -std=c++11

# set compiler 
CC     = mpicc
CXX    = mpicxx

# Default: Build all (main and xbraid)
all: $(BRAID_LIB_FILE) main 

# link main
main: $(OBJ_FILES) 
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJ_FILES) $(BRAID_LIB_FILE)

# build src files
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(INC)
	@$(CXX) $(CXX_FLAGS) -MM $< -MP -MT $@ -MF $(@:.o=.d) $(INC)

# Build xbraid
$(BRAID_LIB_FILE):
	cd xbraid; make braid


.PHONY: all $(BRAID_LIB_FILE) clean cleanall

clean: 
	rm -fr $(BUILD_DIR)
	rm -f main

cleanall: 
	make clean
	rm -f $(BRAID_LIB_FILE)


# include the dependency files
-include $(OBJ_FILES:.o=.d)
