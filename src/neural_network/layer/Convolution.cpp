#include "Convolution.hpp"

namespace snn::internal
{
Convolution::Convolution(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, std::move(optimizer))
{
}

inline auto Convolution::computeOutput(const std::vector<float>& inputs, [[maybe_unused]] bool temporalReset)
    -> std::vector<float>
{
    std::vector<float> outputs(this->numberOfKernels);
    std::vector<float> neuronInputs(this->sizeOfNeuronInputs);
    for (size_t k = 0, o = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (size_t i = 0; i < neuronInputs.size(); ++i)
        {
            const auto& index = kernelIndexes[k][i];
            if (index >= 0)
            {
                [[likely]] neuronInputs[i] = inputs[index];
            }
            else
            {
                [[unlikely]] neuronInputs[i] = 0;
            }
        }
        for (size_t n = 0; n < this->neurons.size(); ++n, ++o)
        {
            outputs[o] = this->neurons[n].output(neuronInputs);
        }
    }
    return outputs;
}

inline auto Convolution::computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float>
{
    std::vector<float> errors(this->numberOfInputs, 0);
    for (size_t k = 0, i = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (auto& neuron : this->neurons)
        {
            auto& error = neuron.backOutput(inputErrors[i]);
            for (size_t e = 0; e < error.size(); ++e)
            {
                const auto& index = kernelIndexes[k][e];
                if (index >= 0)
                {
                    [[likely]] errors[index] += error[e];
                }
            }
            ++i;
        }
    }
    for (int f = 0; f < this->numberOfFilters; ++f)
    {
        this->neurons[f].train();
    }
    return errors;
}

inline void Convolution::computeTrain(std::vector<float>& inputErrors)
{
    for (size_t k = 0, i = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (auto& neuron : this->neurons)
        {
            neuron.back(inputErrors[i]);
            ++i;
        }
    }
    for (int f = 0; f < this->numberOfFilters; ++f)
    {
        this->neurons[f].train();
    }
}
}  // namespace snn::internal
