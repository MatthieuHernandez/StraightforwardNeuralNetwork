#include "NoNeuronLayer.hpp"

void* snn::internal::NoNeuronLayer::getNeuron(int index)
{
    return nullptr;
}

int snn::internal::NoNeuronLayer::getNumberOfNeurons() const
{
    return 0;
}

int snn::internal::NoNeuronLayer::getNumberOfParameters() const
{
    return 0;
}