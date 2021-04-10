#include <boost/serialization/export.hpp>
#include "LocallyConnected1D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LocallyConnected1D)

LocallyConnected1D::LocallyConnected1D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
    const int rest = this->shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->shapeOfInput[0] / this->sizeOfFilterMatrix + rest,
        this->numberOfFilters
    };
    this->sizeOfNeuronInputs = this->sizeOfFilterMatrix * this->shapeOfInput[1];
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
        if (neuron.getNumberOfInputs() != this->sizeOfFilterMatrix * this->shapeOfInput[1])
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
vector<float> LocallyConnected1D::createInputsForNeuron(const int neuronIndex, const vector<float>& inputs)
{
    const int beginIndex = neuronIndex * this->sizeOfNeuronInputs;
    const int endIndex = beginIndex + this->sizeOfNeuronInputs;

    if (endIndex <= this->shapeOfInput[0])
        return vector<float>(inputs.begin() + beginIndex, inputs.begin() + endIndex);
    else
    {
        auto v = vector<float>(inputs.begin() + beginIndex, inputs.begin() + this->shapeOfInput[1]);
        v.resize(this->sizeOfNeuronInputs, 0);
        return v;
    }
}

inline
void LocallyConnected1D::insertBackOutputForNeuron(const int neuronIndex, const std::vector<float>& error, std::vector<float>& errors)
{
    const int beginIndex = neuronIndex * this->sizeOfNeuronInputs;
    for (int e = 0; e < (int)error.size(); ++e)
    {
        const int i = beginIndex + e;
        errors[i] += error[e];
    }
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
