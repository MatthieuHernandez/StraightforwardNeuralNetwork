#pragma once
#include <vector>

#include "../../optimizer/NeuralNetworkOptimizer.hpp"

namespace snn::internal
{
template <class N>
concept HasNonTemporalOuputMethod = requires(N neuron, std::vector<float> inputs) {
    { neuron.output(inputs) } -> std::same_as<float>;
};

template <class N>
concept HasTemporalOuputMethod = requires(N neuron, std::vector<float> inputs) {
    { neuron.output(inputs, true) } -> std::same_as<float>;
};

template <class N>
concept HasCommonMethods = requires(N neuron, float error, std::shared_ptr<NeuralNetworkOptimizer> optimizer) {
    { neuron.backOutput(error) } -> std::same_as<std::vector<float>&>;
    { neuron.back(error) } -> std::same_as<void>;
    { neuron.train() } -> std::same_as<void>;
    { neuron.setOptimizer(optimizer) } -> std::same_as<void>;
};

template <class N>
concept HasCommonConstMethods = requires(const N neuron) {
    { neuron.isValid() } -> std::same_as<errorType>;

    { neuron.getWeights() } -> std::same_as<std::vector<float>>;
    { neuron.getNumberOfParameters() } -> std::same_as<int>;
    { neuron.getNumberOfInputs() } -> std::same_as<int>;

    { neuron.operator==(neuron) } -> std::same_as<bool>;
    { neuron.operator!=(neuron) } -> std::same_as<bool>;
};

template <class N>
concept BaseNeuron =
    HasCommonMethods<N> && HasCommonConstMethods<N> && (HasNonTemporalOuputMethod<N> || HasTemporalOuputMethod<N>);
}  // namespace snn::internal
