#include "Convolution2D.hpp"

#include <boost/serialization/export.hpp>
#include <utility>

#include "LayerModel.hpp"

namespace snn::internal
{
Convolution2D::Convolution2D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, std::move(optimizer))
{
    this->shapeOfOutput = {
        this->numberOfFilters,
        this->shapeOfInput[X] - (this->kernelSize - 1),
        this->shapeOfInput[Y] - (this->kernelSize - 1),
    };
    this->numberOfNeuronsPerFilter = 1;
    this->buildKernelIndexes();
}

void Convolution2D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfKernelsPerFilter);
    const int maxC = this->shapeOfInput[C];
    const int maxX = this->shapeOfInput[X];
    const int kSize = this->kernelSize;
    const int kIndexSize = static_cast<int>(this->kernelIndexes.size());
    for (int k = 0; k < kIndexSize; ++k)
    {
        this->kernelIndexes[k].resize(this->sizeOfNeuronInputs);
        const int kernelPosX = k % this->shapeOfOutput[X];
        const int kernelPosY = k / this->shapeOfOutput[X];
        for (int y = 0; y < kSize; ++y)
        {
            const int inputIndexY = (kernelPosY + y) * maxX + kernelPosX;
            for (int x = 0; x < kSize; ++x)
            {
                const int inputIndexX = (inputIndexY + x) * maxC;
                const int kernelIndexX = (y * kSize + x) * maxC;
                for (int c = 0; c < maxC; ++c)
                {
                    const int inputIndex = inputIndexX + c;
                    const int kernelIndex = kernelIndexX + c;
                    this->kernelIndexes[k][kernelIndex] = inputIndex;
                }
            }
        }
    }
}

inline auto Convolution2D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer>
{
    auto layer = std::make_unique<Convolution2D>(*this);
    for (auto& neuron : layer->neurons)
    {
        neuron.setOptimizer(optimizer);
    }
    return layer;
}

auto Convolution2D::isValid() const -> errorType
{
    for (const auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfNeuronInputs)
        {
            return errorType::conv2DLayerWrongNumberOfInputs;
        }
    }
    return this->FilterLayer::isValid();
}

auto Convolution2D::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " Convolution2D\n";
    summary << "                Input shape:  [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << ", "
            << this->shapeOfInput[2] << "]\n";
    summary << "                Filters:      " << this->numberOfFilters << '\n';
    summary << "                Kernel size:  " << this->kernelSize << "x" << this->kernelSize << '\n';
    summary << "                Parameters:   " << this->getNumberOfParameters() << '\n';
    summary << "                Activation:   " << this->neurons[0].outputFunction->getName() << '\n';
    summary << "                Output shape: [" << this->shapeOfOutput[0] << ", " << this->shapeOfOutput[1] << ", "
            << this->shapeOfOutput[2] << "]\n";
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

inline auto Convolution2D::computeOutput(const std::vector<float>& inputs, [[maybe_unused]] bool temporalReset)
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

inline auto Convolution2D::computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float>
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

inline void Convolution2D::computeTrain(std::vector<float>& inputErrors)
{
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        this->neurons[n].train(inputErrors[n]);
    }
}

inline auto Convolution2D::operator==(const BaseLayer& layer) const -> bool
{
    try
    {
        const auto& f = dynamic_cast<const Convolution2D&>(layer);
        return this->FilterLayer::operator==(layer) && this->sizeOfNeuronInputs == f.sizeOfNeuronInputs;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

inline auto Convolution2D::operator!=(const BaseLayer& layer) const -> bool { return !(*this == layer); }
}  // namespace snn::internal
