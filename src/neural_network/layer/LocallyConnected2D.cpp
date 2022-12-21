#include <boost/serialization/export.hpp>
#include "LocallyConnected2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LocallyConnected2D)

LocallyConnected2D::LocallyConnected2D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
    const int restX = shapeOfInput[0] % this->kernelSize == 0 ? 0 : 1;
    const int restY = shapeOfInput[1] % this->kernelSize == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->shapeOfInput[0] / this->kernelSize + restX,
        this->shapeOfInput[1] / this->kernelSize + restY,
        this->numberOfFilters
    };
    this->numberOfNeuronsPerFilter = this->numberOfKernelsPerFilter;
    this->buildKernelIndexes();
}

void LocallyConnected2D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfKernelsPerFilter);
    const int kSize = this->kernelSize;
    //const int maxX = this->shapeOfInput[0];
    const int maxC = this->shapeOfInput[2];
    for (int k = 0; k < this->kernelIndexes.size(); ++k)
    {
        this->kernelIndexes[k].resize(this->sizeOfNeuronInputs);
        const int kernelPosX = k % this->shapeOfOutput[0];
        const int kernelPosY = k / this->shapeOfOutput[1];
        for (int y = 0; y < kSize; ++y)
        {
            const int inputIndexY = (kernelPosY * kSize + y) * this->shapeOfInput[0] * maxC;

            const int kernelIndexY = y * kSize * maxC;
            for (int x = 0; x < kSize; ++x)
            {
                const int inputIndexX = (kernelPosX * kSize + x) * maxC;
                const int kernelIndexX = x * maxC;
                for (int c = 0; c < maxC; ++c)
                {
                    const int inputIndex = inputIndexY + inputIndexX + c;
                    const int kernelIndex = kernelIndexY + kernelIndexX + c;
                    if (inputIndexX + c < this->shapeOfInput[0] * maxC
                     && inputIndex < this->numberOfInputs)
                        this->kernelIndexes[k][kernelIndex] = inputIndex;
                    else
                        this->kernelIndexes[k][kernelIndex] = -1;
                }
            }
        }
    }
}

inline
unique_ptr<BaseLayer> LocallyConnected2D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<LocallyConnected2D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

int LocallyConnected2D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfNeuronInputs)
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
vector<float> LocallyConnected2D::computeOutput(const vector<float>& inputs, [[maybe_unused]] bool temporalReset)
{
    vector<float> outputs(this->numberOfKernels);
    vector<float> neuronInputs(this->sizeOfNeuronInputs);
    for (size_t k = 0, o = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (size_t i = 0; i < neuronInputs.size(); ++i)
        {
            const auto& index = this->kernelIndexes[k][i];
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
vector<float> LocallyConnected2D::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        auto k = n / this->numberOfFilters;
        for (size_t e = 0; e < error.size(); ++e)
        {
            const auto& index = kernelIndexes[k][e];
            if (index >= 0) [[likely]]
                errors[index] += error[e];
        }
    }
    return errors;
}

inline
void LocallyConnected2D::computeTrain(std::vector<float>& inputErrors)
{
    for (int n = 0; n < this->neurons.size(); ++n)
        this->neurons[n].train(inputErrors[n]);
}

inline
bool LocallyConnected2D::operator==(const BaseLayer& layer) const
{
    try
    {
        const auto& f = dynamic_cast<const LocallyConnected2D&>(layer);
        return this->FilterLayer::operator==(layer)
            && this->sizeOfNeuronInputs == f.sizeOfNeuronInputs;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

inline
bool LocallyConnected2D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
