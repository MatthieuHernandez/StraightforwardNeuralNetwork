#pragma once
#include "SimpleLayer.hpp"

namespace snn::internal
{
template <BaseNeuron N>
SimpleLayer<N>::SimpleLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Layer<N>(model, optimizer)
{
}

template <BaseNeuron N>
auto SimpleLayer<N>::computeOutput(const std::vector<float>& inputs, [[maybe_unused]] bool temporalReset)
    -> std::vector<float>
{
    std::vector<float> outputs(this->neurons.size());
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs);
    }
    return outputs;
}

template <BaseNeuron N>
auto SimpleLayer<N>::computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float>
{
    std::vector<float> errors(this->numberOfInputs, 0);
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        this->neurons[n].train();
        for (size_t e = 0; e < errors.size(); ++e)
        {
            errors[e] += error[e];
        }
    }
    return errors;
}

template <BaseNeuron N>
void SimpleLayer<N>::computeTrain(std::vector<float>& inputErrors)
{
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        this->neurons[n].back(inputErrors[n]);
        this->neurons[n].train();
    }
}

template <BaseNeuron N>
std::vector<int> SimpleLayer<N>::getShapeOfInput() const
{
    return {this->getNumberOfInputs()};
}

template <BaseNeuron N>
std::vector<int> SimpleLayer<N>::getShapeOfOutput() const
{
    return {this->getNumberOfNeurons()};
}

template <BaseNeuron N>
auto SimpleLayer<N>::isValid() const -> errorType
{
    int numberOfOutput = 1;
    auto shape = this->getShapeOfOutput();
    for (const int s : shape)
    {
        numberOfOutput *= s;
    }

    if (numberOfOutput != this->getNumberOfNeurons())
    {
        return errorType::layerWrongNumberOfOutputs;
    }

    for (auto& neuron : this->neurons)
    {
        if (neuron.getNumberOfInputs() != this->getNumberOfInputs())
        {
            return errorType::layerWrongNumberOfInputs;
        }
    }
    return Layer<N>::isValid();
}

template <BaseNeuron N>
auto SimpleLayer<N>::operator==(const BaseLayer& layer) const -> bool
{
    return Layer<N>::operator==(layer);
}

template <BaseNeuron N>
auto SimpleLayer<N>::operator!=(const BaseLayer& layer) const -> bool
{
    return !(*this == layer);
}
}  // namespace snn::internal
