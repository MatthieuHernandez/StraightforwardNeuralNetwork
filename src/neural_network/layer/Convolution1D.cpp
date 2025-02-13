#include "Convolution1D.hpp"

#include <boost/serialization/export.hpp>
#include <utility>

#include "LayerModel.hpp"

namespace snn::internal
{
Convolution1D::Convolution1D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, std::move(optimizer))
{
    this->shapeOfOutput = {
        this->numberOfFilters,
        this->shapeOfInput[X] - (this->kernelSize - 1),
    };
    this->numberOfNeuronsPerFilter = 1;
    this->buildKernelIndexes();
}

void Convolution1D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfKernelsPerFilter);
    const int maxC = this->shapeOfInput[C];
    const int kSize = this->kernelSize;
    const int kIndexSize = static_cast<int>(this->kernelIndexes.size());
    for (int k = 0; k < kIndexSize; ++k)
    {
        this->kernelIndexes[k].resize(this->sizeOfNeuronInputs);
        for (int x = 0; x < kSize; ++x)
        {
            const int inputIndexX = (k + x) * maxC;
            const int kernelIndexX = x * maxC;
            for (int c = 0; c < maxC; ++c)
            {
                const int inputIndex = inputIndexX + c;
                const int kernelIndex = kernelIndexX + c;
                this->kernelIndexes[k][kernelIndex] = inputIndex;
            }
        }
    }
}

inline auto Convolution1D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer>
{
    auto layer = std::make_unique<Convolution1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto Convolution1D::isValid() const -> errorType
{
    for (const auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->kernelSize * this->shapeOfInput[C])
        {
            return errorType::conv1DLayerWrongNumberOfInputs;
        }
    }
    return this->FilterLayer::isValid();
}

auto Convolution1D::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " Convolution1D\n";
    summary << "                Input shape:  [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << "]\n";
    summary << "                Filters:      " << this->numberOfFilters << '\n';
    summary << "                Kernel size:  " << this->kernelSize << '\n';
    summary << "                Parameters:   " << this->getNumberOfParameters() << '\n';
    summary << "                Activation:   " << this->neurons[0].outputFunction->getName() << '\n';
    summary << "                Output shape: [" << this->shapeOfOutput[0] << ", " << this->shapeOfOutput[1] << "]"
            << '\n';
    if (!optimizers.empty())
    {
        summary << "                Optimizers:   " << optimizers[0]->summary() << '\n';
    }
    for (size_t o = 1; o < this->optimizers.size(); ++o)
    {
        summary << "                              " << optimizers[o]->summary() << '\n';
    }
    return summary.str();
}

inline auto Convolution1D::computeOutput(const std::vector<float>& inputs, [[maybe_unused]] bool temporalReset)
    -> std::vector<float>
{
    std::vector<float> outputs(this->numberOfKernels);
    std::vector<float> neuronInputs(this->sizeOfNeuronInputs);
    for (size_t k = 0, o = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (size_t i = 0; i < neuronInputs.size(); ++i)
        {
            const auto& index = kernelIndexes[k][i];
            neuronInputs[i] = inputs[index];
        }
        for (size_t n = 0; n < this->neurons.size(); ++n, ++o)
        {
            outputs[o] = this->neurons[n].output(neuronInputs);
        }
    }
    return outputs;
}

inline auto Convolution1D::computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float>
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
                errors[index] += error[e];
            }
            ++i;
        }
    }
    return errors;
}

inline void Convolution1D::computeTrain(std::vector<float>& inputErrors)
{
    for (int n = 0; n < this->numberOfFilters; ++n)
    {
        this->neurons[n].train(inputErrors[n]);
    }
}

inline auto Convolution1D::operator==(const BaseLayer& layer) const -> bool
{
    return this->FilterLayer::operator==(layer);
}

inline auto Convolution1D::operator!=(const BaseLayer& layer) const -> bool { return !(*this == layer); }
}  // namespace snn::internal
