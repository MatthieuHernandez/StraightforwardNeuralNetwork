SOURCES = src/tools/Tools.cpp \
    src/data/Data.cpp \
    src/data/DataForClassification.cpp \
    src/data/DataForMultipleClassification.cpp \
    src/data/DataForRegression.cpp \
    src/neural_network/NeuralNetwork.cpp \
    src/neural_network/NeuralNetworkGettersAndSetters.cpp \
    src/neural_network/StraightforwardOption.cpp \
    src/neural_network/layer/AllToAll.cpp \
    src/neural_network/layer/Layer.cpp \
    src/neural_network/layer/perceptron/Perceptron.cpp \
    src/neural_network/layer/perceptron/activation_function/ActivationFunction.cpp

OBJECTS = $(SOURCES:.cpp=.o)

# Flags passed to the C++ compiler.
CXXFLAGS += -g

all: StraightforwardNeuralNetwork

StraightforwardNeuralNetwork: $(OBJECTS)