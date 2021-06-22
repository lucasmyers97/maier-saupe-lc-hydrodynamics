# General Makefile for any project -- just change TARGET_EXEC
TARGET_EXEC := dealii_function_test

# Define home directory based on shell environment variable
HOME_DIR := $(shell echo $$MAIER_SAUPE_DIR)

# Specify relevant directories relative to home
SRC_DIRS := $(HOME_DIR)/src
EXT_SRC_DIRS := $(HOME_DIR)/extern-src
INC_DIR := $(HOME_DIR)/include

# Find all source files
SRCS := $(shell cd $(SRC_DIRS) ; \
	find -name '*.cpp' -or -name '*.c')
SRCS := $(SRCS:./%=$(SRC_DIRS)/%)

EXT_SRCS := $(shell cd $(EXT_SRC_DIRS) ; \
	find -name '*.cpp' -or -name '*.c')
EXT_SRCS := $(EXT_SRCS:./%=$(EXT_SRC_DIRS)/%)

INT_SRCS := $(shell find -name '*.cpp' -or -name '*.c')

# Specify object file names
# First line commented out b/c it contains references to dealii code
# OBJS := $(SRCS:%.cpp=%.o)
OBJS += $(EXT_SRCS:%.cpp=%.o)
OBJS += $(INT_SRCS:%.cpp=%.o)

# Specify where header files are
INC_DIRS := $(shell cd $(INC_DIR) ; \
	find . -type d)
INC_DIRS := $(INC_DIRS:.%=$(INC_DIR)/%)
INC_FLAGS := $(addprefix -I ,$(INC_DIRS))

# Link final executable
$(TARGET_EXEC) : $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Compile all source files
$(SRC_DIRS)/%.o : %.cpp
	@-echo $<
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@ $(INC_FLAGS)

$(EXT_SRC_DIRS)/%.o : %.cpp 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@ $(INC_FLAGS)

./%.o : %.cpp 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@ $(INC_FLAGS)


.PHONY : clean
clean :
	rm -r $(INT_SRCS:%.cpp=%.o)
