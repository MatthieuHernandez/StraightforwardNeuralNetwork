
# Points to the root of fused Google Test, relative to where this file is.
GTEST_DIR = googletest-release-1.10.0/googletest

SOURCES = unit_tests/main.cpp \
    unit_tests/SaveNeuralNetworkTest.cpp \
    unit_tests/UnitTestTests.cpp \

PROJECT_DIR = ..

#VPATH += :$(GTEST_DIR):$(PROJECT_DIR)
# Paths to the fused gtest files.
#GTEST_H = $(GTEST_DIR)/include/gtest/gtest.h
#GTEST_ALL_CC = $(GTEST_DIR)/src/gtest-all.cc
#GTEST_MAIN_CC = $(GTEST_DIR)/src/gtest_main.cc

OBJECTS = $(SOURCES:.cpp=.o)

#subsystem: 
#	$(MAKE) -C $(PROJECT_DIR)
#
# variable BOOST_DIR

BOOST_DIR = $(PROJECT_DIR)/src/external_library/boost_1_71_0/

CXXFLAGS += -g -std=c++17 -I $(PROJECT_DIR)/src -I $(GTEST_DIR)/include -I $(GTEST_DIR)/include -I $(BOOST_DIR)

all: Step1 \
	BuildProject \
	Step2 \
	BuildGoogleTest \
	Step3 \
	BuildUnitTests \

Step1: ;@echo "Step 1 : Build project"
BuildProject: ;$(MAKE) -C $(PROJECT_DIR)
Step2: ;@echo "Step 2 : Build GoogleTest library"
BuildGoogleTest: ;$(MAKE) -C $(GTEST_DIR)
Step3: ;@echo "Step 3 : Build unit test"
BuildUnitTests: $(OBJECTS) 