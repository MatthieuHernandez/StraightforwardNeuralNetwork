#include "NoNeuronLayer.hpp"

#include <boost/serialization/export.hpp>

using namespace std;
using namespace snn;
using namespace internal;

int NoNeuronLayer::getNumberOfOutput() const { return this->numberOfOutputs; }

float NoNeuronLayer::getAverageOfAbsNeuronWeights() const { return 0.0f; }

float NoNeuronLayer::getAverageOfSquareNeuronWeights() const { return 0.0f; }

void* NoNeuronLayer::getNeuron([[maybe_unused]] int index) { return nullptr; }

int NoNeuronLayer::getNumberOfNeurons() const { return 0; }

int NoNeuronLayer::getNumberOfParameters() const { return 0; }