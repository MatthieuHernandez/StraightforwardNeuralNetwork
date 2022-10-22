#pragma once
#include <vector>
#include "../../optimizer/NeuralNetworkOptimizer.hpp"
#include "../../../tools/Tensor.hpp"

namespace snn::internal
{
    template <class N>
    concept HasNonTemporalOuputMethod =
    requires(N neuron, Tensor inputs)
    {
        { neuron.output(inputs) } -> std::same_as<float>;
    };

    template <class N>
    concept HasTemporalOuputMethod =
    requires(N neuron, Tensor inputs)
    {
        { neuron.output(inputs, true) } -> std::same_as<float>;
    };

    template <class N>
    concept HasCommonMethods =
    requires(N neuron, float error, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    {
        { neuron.backOutput(error) } -> std::same_as<Tensor&>;
        { neuron.train(error) } -> std::same_as<void>;
        { neuron.setOptimizer(optimizer) } -> std::same_as<void>;
    };

    template <class N>
    concept HasCommonConstMethods =
    requires(const N neuron)
    {
        { neuron.isValid() } -> std::same_as<int>;

        { neuron.getWeights() } -> std::same_as<Tensor>;
        { neuron.getNumberOfParameters() } -> std::same_as<int>;
        { neuron.getNumberOfInputs() } -> std::same_as<int>;

        { neuron.getNumberOfInputs() } -> std::same_as<int>;
        { neuron.operator==(neuron) } -> std::same_as<bool>;
        { neuron.operator!=(neuron) } -> std::same_as<bool>;
    };

    template <class N>
    concept BaseNeuron = HasCommonMethods<N> && HasCommonConstMethods<N> && (HasNonTemporalOuputMethod<N> || HasTemporalOuputMethod<N>);
}
