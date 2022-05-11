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
    this->sizeOfNeuronInputs = this->kernelSize * this->shapeOfInput[1];
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
    vector<float> outputs(this->neurons.size());
    for (int i = 0, n = 0; n < (int)this->neurons.size(); ++i)
    {
        const int beginIndex = i * this->sizeOfNeuronInputs;
        const int endIndex = beginIndex + this->sizeOfNeuronInputs;
        vector<float> neuronInputs;
        if (endIndex <= this->shapeOfInput[0])
            neuronInputs = vector<float>{inputs.begin() + beginIndex, inputs.begin() + endIndex};
        else
        {
            neuronInputs = vector(inputs.begin() + beginIndex, inputs.begin() + this->shapeOfInput[1]);
            neuronInputs.resize(this->sizeOfNeuronInputs, 0);
        }
        for (int f = 0; f < this->numberOfFilters; ++f, ++n)
        {
            outputs[n] = this->neurons[n].output(neuronInputs);
        }
    }
    return outputs;
}

inline
vector<float> LocallyConnected1D::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < (int)this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        const int neuronIndex = n / this->numberOfFilters;
        const int beginIndex = neuronIndex * this->sizeOfNeuronInputs;
        for (int e = 0; e < (int)error.size(); ++e)
        {
            const int i = beginIndex + e;
            errors[i] += error[e];
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
