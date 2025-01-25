#include "NoNeuronLayer.hpp"

#include <boost/serialization/export.hpp>

using namespace std;
using namespace snn;
using namespace internal;

auto NoNeuronLayer::getNumberOfOutput() const -> int { return this->numberOfOutputs; }

auto NoNeuronLayer::getAverageOfAbsNeuronWeights() const -> float { return 0.0F; }

auto NoNeuronLayer::getAverageOfSquareNeuronWeights() const -> float { return 0.0F; }

auto NoNeuronLayer::getNeuron([[maybe_unused]] int index) -> void* { return nullptr; }

auto NoNeuronLayer::getNumberOfNeurons() const -> int { return 0; }

auto NoNeuronLayer::getNumberOfParameters() const -> int { return 0; }