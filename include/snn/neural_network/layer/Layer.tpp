#pragma once
#include "../optimizer/LayerOptimizerFactory.hpp"
#include "Layer.hpp"

namespace snn::internal
{
template <BaseNeuron N>
Layer<N>::Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : numberOfInputs(model.numberOfInputs)
{
    this->neurons.reserve(model.numberOfNeurons);
    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(model.neuron, optimizer);
    }
    LayerOptimizerFactory::build(this->optimizers, model, this);
}

template <BaseNeuron N>
Layer<N>::Layer(const Layer& layer)
    : numberOfInputs(layer.numberOfInputs),
      neurons(layer.neurons)
{
    this->optimizers.reserve(layer.optimizers.size());
    for (const auto& optimizer : layer.optimizers)
    {
        this->optimizers.emplace_back(optimizer->clone(this));
    }
}

template <BaseNeuron N>
std::vector<float> Layer<N>::output(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = this->computeOutput(inputs, temporalReset);
    for (auto& optimizer : this->optimizers)
    {
        optimizer->applyAfterOutputForTesting(output);
    }
    return output;
}

template <BaseNeuron N>
std::vector<float> Layer<N>::outputForTraining(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = this->computeOutput(inputs, temporalReset);
    for (auto& optimizer : this->optimizers)
    {
        optimizer->applyAfterOutputForTraining(output, temporalReset);
    }
    return output;
}

template <BaseNeuron N>
std::vector<float> Layer<N>::backOutput(std::vector<float>& inputErrors)
{
    for (auto& optimizer : this->optimizers)
    {
        optimizer->applyBeforeBackpropagation(inputErrors);
    }
    auto error = this->computeBackOutput(inputErrors);
    return error;
}

template <BaseNeuron N>
void Layer<N>::train(std::vector<float>& inputErrors)
{
    for (auto& optimizer : this->optimizers)
    {
        optimizer->applyBeforeBackpropagation(inputErrors);
    }
    this->computeTrain(inputErrors);
}

template <BaseNeuron N>
auto Layer<N>::isValid() const -> errorType
{
    if (this->getNumberOfNeurons() != static_cast<int>(this->neurons.size()) || this->getNumberOfNeurons() < 1 ||
        this->getNumberOfNeurons() > 1000000)
    {
        return errorType::layerTooMuchNeurons;
    }

    for (auto& neuron : this->neurons)
    {
        const auto err = neuron.isValid();
        if (err != errorType::noError)
        {
            return err;
        }
    }
    return errorType::noError;
}

template <BaseNeuron N>
auto Layer<N>::getNeuron(int index) -> void*
{
    return static_cast<void*>(&this->neurons.at(index));
}

template <BaseNeuron N>
auto Layer<N>::getAverageOfAbsNeuronWeights() const -> float
{
    auto sum = 0.0F;
    for (auto& n : this->neurons)
    {
        for (auto w : n.getWeights())
        {
            sum += std::abs(w);
        }
    }
    sum /= static_cast<float>(this->neurons.size());
    return sum;
}

template <BaseNeuron N>
auto Layer<N>::getAverageOfSquareNeuronWeights() const -> float
{
    auto sum = 0.0F;
    for (auto& n : this->neurons)
    {
        for (auto w : n.getWeights())
        {
            sum += w * w;
        }
    }
    sum /= static_cast<float>(this->neurons.size());
    return sum;
}

template <BaseNeuron N>
auto Layer<N>::getNumberOfInputs() const -> int
{
    return this->numberOfInputs;
}

template <BaseNeuron N>
auto Layer<N>::getNumberOfNeurons() const -> int
{
    return static_cast<int>(this->neurons.size());
}

template <BaseNeuron N>
auto Layer<N>::getNumberOfParameters() const -> int
{
    int sum = 0;
    for (auto& neuron : this->neurons)
    {
        sum += neuron.getNumberOfParameters();
    }
    return sum;
}

template <BaseNeuron N>
auto Layer<N>::operator==(const BaseLayer& layer) const -> bool
{
    try
    {
        const auto& l = dynamic_cast<const Layer&>(layer);

        return typeid(*this).hash_code() == typeid(layer).hash_code() && this->numberOfInputs == l.numberOfInputs &&
               this->neurons == l.neurons && [this, &l]()
        {
            for (size_t o = 0; o < this->optimizers.size(); ++o)
            {
                if (*this->optimizers[o] != *l.optimizers[o])
                {
                    return false;
                }
            }
            return true;
        }();
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

template <BaseNeuron N>
auto Layer<N>::operator!=(const BaseLayer& layer) const -> bool
{
    return !(*this == layer);
}
}  // namespace snn::internal
