#include "LocallyConnected2D.hpp"

#include <boost/serialization/export.hpp>

#include "LayerModel.hpp"

namespace snn::internal
{
LocallyConnected2D::LocallyConnected2D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
    const int restX = shapeOfInput[X] % this->kernelSize == 0 ? 0 : 1;
    const int restY = shapeOfInput[Y] % this->kernelSize == 0 ? 0 : 1;

    this->shapeOfOutput = {this->numberOfFilters, this->shapeOfInput[X] / this->kernelSize + restX,
                           this->shapeOfInput[Y] / this->kernelSize + restY};
    this->numberOfNeuronsPerFilter = this->numberOfKernelsPerFilter;
    this->buildKernelIndexes();
}

void LocallyConnected2D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfKernelsPerFilter);
    const int kSize = this->kernelSize;
    const int maxC = this->shapeOfInput[C];
    const int kIndexSize = static_cast<int>(this->kernelIndexes.size());
    for (int k = 0; k < kIndexSize; ++k)
    {
        this->kernelIndexes[k].resize(this->sizeOfNeuronInputs);
        const int kernelPosX = k % this->shapeOfOutput[X];
        const int kernelPosY = k / this->shapeOfOutput[Y];
        for (int y = 0; y < kSize; ++y)
        {
            const int inputIndexY = (kernelPosY * kSize + y) * this->shapeOfInput[X] * maxC;

            const int kernelIndexY = y * kSize * maxC;
            for (int x = 0; x < kSize; ++x)
            {
                const int inputIndexX = (kernelPosX * kSize + x) * maxC;
                const int kernelIndexX = x * maxC;
                for (int c = 0; c < maxC; ++c)
                {
                    const int inputIndex = inputIndexY + inputIndexX + c;
                    const int kernelIndex = kernelIndexY + kernelIndexX + c;
                    if (inputIndexX + c < this->shapeOfInput[X] * maxC && inputIndex < this->numberOfInputs)
                    {
                        this->kernelIndexes[k][kernelIndex] = inputIndex;
                    }
                    else
                    {
                        this->kernelIndexes[k][kernelIndex] = -1;
                    }
                }
            }
        }
    }
}

inline auto LocallyConnected2D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
    -> std::unique_ptr<BaseLayer>
{
    auto layer = std::make_unique<LocallyConnected2D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto LocallyConnected2D::isValid() const -> errorType
{
    for (const auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfNeuronInputs)
        {
            return errorType::locallyConnected2DWrongNumberOfInputs;
        }
    }
    return this->FilterLayer::isValid();
}

auto LocallyConnected2D::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " LocallyConnected2D";
    summary << "                Input shape: [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << ", "
            << this->shapeOfInput[2] << "]\n";
    summary << "                Filters: " << this->numberOfFilters;
    summary << "                Kernel size: " << this->kernelSize << "x" << this->kernelSize << '\n';
    summary << "                Parameters: " << this->getNumberOfParameters() << '\n';
    summary << "                Activation: " << this->neurons[0].outputFunction->getName() << '\n';
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

inline auto LocallyConnected2D::computeOutput(const std::vector<float>& inputs, [[maybe_unused]] bool temporalReset)
    -> std::vector<float>
{
    std::vector<float> outputs(this->numberOfKernels);
    std::vector<float> neuronInputs(this->sizeOfNeuronInputs);
    for (size_t k = 0, o = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (size_t i = 0; i < neuronInputs.size(); ++i)
        {
            const auto& index = this->kernelIndexes[k][i];
            if (index >= 0)
            {
                [[likely]] neuronInputs[i] = inputs[index];
            }
            else
            {
                [[unlikely]] neuronInputs[i] = 0;
            }
        }
        for (int n = 0; n < this->numberOfFilters; ++n, ++o)
        {
            outputs[o] = this->neurons[o].output(neuronInputs);
        }
    }
    return outputs;
}

inline auto LocallyConnected2D::computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float>
{
    std::vector<float> errors(this->numberOfInputs, 0);
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        this->neurons[n].train();
        auto k = n / this->numberOfFilters;
        for (size_t e = 0; e < error.size(); ++e)
        {
            const auto& index = kernelIndexes[k][e];
            if (index >= 0)
            {
                [[likely]] errors[index] += error[e];
            }
        }
    }
    return errors;
}

inline void LocallyConnected2D::computeTrain(std::vector<float>& inputErrors)
{
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        this->neurons[n].back(inputErrors[n]);
        this->neurons[n].train();
    }
}

inline auto LocallyConnected2D::operator==(const BaseLayer& layer) const -> bool
{
    try
    {
        const auto& f = dynamic_cast<const LocallyConnected2D&>(layer);
        return this->FilterLayer::operator==(layer) && this->sizeOfNeuronInputs == f.sizeOfNeuronInputs;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}
}  // namespace snn::internal
