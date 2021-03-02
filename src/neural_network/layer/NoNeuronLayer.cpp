#include <boost/serialization/export.hpp>
#include "NoNeuronLayer.hpp"

using namespace std;
using namespace snn;
using namespace internal;


int NoNeuronLayer::getNumberOfOutput() const
{
    return numberOfOutputs;
}

float NoNeuronLayer::getSumOfAbsNeuronWeights() const
{
    return 0.0f;
}

float NoNeuronLayer::getSumOfSquareNeuronWeights() const
{
    return 0.0f;
}

void* NoNeuronLayer::getNeuron(int index)
{
    return nullptr;
}

int NoNeuronLayer::getNumberOfNeurons() const
{
    return 0;
}

int NoNeuronLayer::getNumberOfParameters() const
{
    return 0;
}