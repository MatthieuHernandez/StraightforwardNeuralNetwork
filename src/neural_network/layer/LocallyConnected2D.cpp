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
    this->sizeOfNeuronInputs = this->kernelSize * this->kernelSize * this->shapeOfInput[2];
    this->neuronInputs.resize(this->sizeOfNeuronInputs);
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
    vector<float> outputs(this->neurons.size());
    for (int p = 0, n = 0; n < (int)this->neurons.size(); ++p)
    {
        const int neuronPosX = p % this->shapeOfOutput[0] * this->kernelSize;
        const int neuronPosY = p / this->shapeOfOutput[0] * this->kernelSize;

        for (int i = 0; i < this->kernelSize; ++i)
        {
            const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] + neuronPosX) * this->shapeOfInput[2];
            for (int j = 0; j < this->kernelSize; ++j)
            {
                for (int k = 0; k < this->shapeOfInput[2]; ++k)
                {
                    const int indexErrors = beginIndex + (j * this->shapeOfInput[2] + k);
                    const int indexKernel = (i * this->kernelSize + j) * this->shapeOfInput[2] + k;
                    if (indexErrors < (int)inputs.size())
                        [[likely]]
                            this->neuronInputs[indexKernel] = inputs[indexErrors];
                    else
                        neuronInputs[indexKernel] = 0.0f;
                }
            }
        }
        for (int f = 0; f < this->numberOfFilters; ++f, ++n)
        {
            outputs[n] = this->neurons[n].output(neuronInputs);
        }
    }
    return outputs;
}

inline
vector<float> LocallyConnected2D::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < (int)this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        const int neuronIndex = n / this->numberOfFilters;
        const int neuronPosX = neuronIndex % this->shapeOfOutput[0] * this->kernelSize;
        const int neuronPosY = neuronIndex / this->shapeOfOutput[0] * this->kernelSize;

        for (int i = 0; i < this->kernelSize; ++i)
        {
            const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] + neuronPosX) * this->shapeOfInput[2];
            for (int j = 0; j < this->kernelSize; ++j)
            {
                for (int k = 0; k < this->shapeOfInput[2]; ++k)
                {
                    const int indexErrors = beginIndex + (j * this->shapeOfInput[2] + k);
                    const int indexKernel = (i * this->kernelSize + j) * this->shapeOfInput[2] + k;
                    if (indexErrors < (int)errors.size())
                        [[likely]]
                            errors[indexErrors] += error[indexKernel];
                }
            }
        }
    }
    return errors;
}

inline
void LocallyConnected2D::computeTrain(std::vector<float>& inputErrors)
{
    for (size_t n = 0; n < this->neurons.size(); ++n)
        this->neurons[n].train(inputErrors[n]);
}

inline
bool LocallyConnected2D::operator==(const BaseLayer& layer) const
{
    try
    {
        const auto& f = dynamic_cast<const LocallyConnected2D&>(layer);
        return this->FilterLayer::operator==(layer)
            && this->sizeOfNeuronInputs == f.sizeOfNeuronInputs
            && this->neuronInputs == f.neuronInputs;
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
