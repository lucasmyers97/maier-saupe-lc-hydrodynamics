# General Makefile for any project -- just change TARGET_EXEC
TARGET_EXEC := lebedev_test

HOME_DIR := $(shell echo $$MAIER_SAUPE_DIR)

BUILD_DIR := ./build
SRC_DIRS := $(HOME_DIR)/src
EXT_SRC_DIRS := $(HOME_DIR)/extern-src
INC_DIR := $(HOME_DIR)/include

SRCS := $(shell cd $(SRC_DIRS) ; \
	find -name '*.cpp' -or -name '*.c')
SRCS := $(SRCS:./%=$(SRC_DIRS)/%)

EXT_SRCS := $(shell cd $(EXT_SRC_DIRS) ; \
	find -name '*.cpp' -or -name '*.c')
EXT_SRCS := $(EXT_SRCS:./%=$(EXT_SRC_DIRS)/%)

INT_SRCS := $(shell find -name '*.cpp' -or -name '*.c')

# OBJS := $(SRCS:%.cpp=%.o)
OBJS += $(EXT_SRCS:%.cpp=%.o)
OBJS += $(INT_SRCS:%.cpp=%.o)

INC_DIRS := $(shell cd $(INC_DIR) ; \
	find . -type d)
INC_DIRS := $(INC_DIRS:.%=$(INC_DIR)/%)
INC_FLAGS := $(addprefix -I ,$(INC_DIRS))

$(TARGET_EXEC) : $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

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

print_homedir :
	echo $(HOME_DIR)

print_src :
	@-echo $(SRC_DIRS)
	@-echo $(SRCS)
	@-echo $(EXT_SRCS)
	@-echo $(INT_SRCS)

print_cc :
	@-echo $(CC)
	@-echo $(LDFLAGS)
	@-echo $(dir $@)

print_include :
	@-echo $(INC_FLAGS)

print_compiler :
	@-echo $(CXX)
	@-echo $(CC)
