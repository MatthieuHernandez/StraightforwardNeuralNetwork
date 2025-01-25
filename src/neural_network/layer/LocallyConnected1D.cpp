#include "LocallyConnected1D.hpp"

#include <boost/serialization/export.hpp>
#include <utility>

#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

LocallyConnected1D::LocallyConnected1D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, std::move(optimizer))
{
    const int rest = this->shapeOfInput[X] % this->kernelSize == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->numberOfFilters,
        this->shapeOfInput[X] / this->kernelSize + rest,
    };
    this->numberOfNeuronsPerFilter = this->numberOfKernelsPerFilter;
    this->buildKernelIndexes();
}

void LocallyConnected1D::buildKernelIndexes()
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
            const int inputIndexX = (k * kSize + x) * maxC;
            const int kernelIndexX = x * maxC;
            for (int c = 0; c < maxC; ++c)
            {
                const int inputIndex = inputIndexX + c;
                const int kernelIndex = kernelIndexX + c;
                if (inputIndex < this->numberOfInputs)
                    this->kernelIndexes[k][kernelIndex] = inputIndex;
                else
                    this->kernelIndexes[k][kernelIndex] = -1;
            }
        }
    }
}

inline auto LocallyConnected1D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> unique_ptr<BaseLayer>
{
    auto layer = make_unique<LocallyConnected1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto LocallyConnected1D::isValid() const -> ErrorType
{
    for (const auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->kernelSize * this->shapeOfInput[C])
        {
            return ErrorType::locallyConnected1DWrongNumberOfInputs;
        }
    }
    return this->FilterLayer::isValid();
}

auto LocallyConnected1D::summary() const -> std::string
{
    stringstream ss;
    ss << "------------------------------------------------------------" << endl;
    ss << " LocallyConnected1D";
    ss << "                Input shape: [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << "]" << endl;
    ss << "                Filters: " << this->numberOfFilters << endl;
    ss << "                Kernel size: " << this->kernelSize << endl;
    ss << "                Parameters: " << this->getNumberOfParameters() << endl;
    ss << "                Activation: " << this->neurons[0].outputFunction->getName() << endl;
    ss << "                Output shape: [" << this->shapeOfOutput[0] << ", " << this->shapeOfOutput[1] << "]" << endl;
    if (!optimizers.empty())
    {
        ss << "                Optimizers:   " << optimizers[0]->summary() << endl;
    }
    for (size_t o = 1; o < this->optimizers.size(); ++o)
    {
        ss << "                              " << optimizers[o]->summary() << endl;
    }
    return ss.str();
}

inline auto LocallyConnected1D::computeOutput(const vector<float>& inputs, [[maybe_unused]] bool temporalReset)
    -> vector<float>
{
    vector<float> outputs(this->numberOfKernels);
    vector<float> neuronInputs(this->sizeOfNeuronInputs);
    for (size_t k = 0, o = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (size_t i = 0; i < neuronInputs.size(); ++i)
        {
            const auto& index = kernelIndexes[k][i];
            if (index >= 0) [[likely]]
                neuronInputs[i] = inputs[index];
            else [[unlikely]]
                neuronInputs[i] = 0;
        }
        for (int n = 0; n < this->numberOfFilters; ++n, ++o)
        {
            outputs[o] = this->neurons[o].output(neuronInputs);
        }
    }
    return outputs;
}

inline auto LocallyConnected1D::computeBackOutput(vector<float>& inputErrors) -> vector<float>
{
    vector<float> errors(this->numberOfInputs, 0);
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        for (size_t e = 0; e < error.size(); ++e)
        {
            const auto& index = kernelIndexes[n % numberOfKernelsPerFilter][e];
            if (index >= 0) [[likely]]
                errors[index] += error[e];
        }
    }
    return errors;
}

inline void LocallyConnected1D::computeTrain(std::vector<float>& inputErrors)
{
    for (size_t n = 0; n < this->neurons.size(); ++n) this->neurons[n].train(inputErrors[n]);
}

inline auto LocallyConnected1D::operator==(const BaseLayer& layer) const -> bool
{
    return this->FilterLayer::operator==(layer);
}

inline auto LocallyConnected1D::operator!=(const BaseLayer& layer) const -> bool { return !(*this == layer); }
