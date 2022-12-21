#include <boost/serialization/export.hpp>
#include <utility>
#include "LocallyConnected1D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LocallyConnected1D)

LocallyConnected1D::LocallyConnected1D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, std::move(optimizer))
{
    const int rest = this->shapeOfInput[0] % this->kernelSize == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->shapeOfInput[0] / this->kernelSize + rest,
        this->numberOfFilters
    };
    this->numberOfNeuronsPerFilter = this->numberOfKernelsPerFilter;
    this->buildKernelIndexes();
}

void LocallyConnected1D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfKernelsPerFilter);
    const int maxC = this->shapeOfInput[1];
    const int kSize = this->kernelSize;
    for (int k = 0; k < this->kernelIndexes.size(); ++k)
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

inline
unique_ptr<BaseLayer> LocallyConnected1D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<LocallyConnected1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

int LocallyConnected1D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->kernelSize * this->shapeOfInput[1])
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
vector<float> LocallyConnected1D::computeOutput(const vector<float>& inputs, [[maybe_unused]] bool temporalReset)
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
        for (size_t n = 0; n < this->numberOfFilters; ++n, ++o)
        {
            outputs[o] = this->neurons[o].output(neuronInputs);
        }
    }
    return outputs;
}

inline
vector<float> LocallyConnected1D::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
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

inline
void LocallyConnected1D::computeTrain(std::vector<float>& inputErrors)
{
    for (size_t n = 0; n < this->neurons.size(); ++n)
        this->neurons[n].train(inputErrors[n]);
}


inline
bool LocallyConnected1D::operator==(const BaseLayer& layer) const
{
    return this->FilterLayer::operator==(layer);
}

inline
bool LocallyConnected1D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
