#include <boost/serialization/export.hpp>
#include "LocallyConnected1D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LocallyConnected1D)

LocallyConnected1D::LocallyConnected1D(LayerModel& model, StochasticGradientDescent* optimizer)
    : Filter(model, optimizer)
{
}

inline
unique_ptr<Layer> LocallyConnected1D::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<LocallyConnected1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

std::vector<int> LocallyConnected1D::getShapeOfOutput() const
{
    const int rest = this->shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    return {
        this->shapeOfInput[0] / this->sizeOfFilterMatrix + rest,
        this->numberOfFilters
    };
}

int LocallyConnected1D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfFilterMatrix * this->shapeOfInput[1])
            return 203;
    }
    return this->Filter::isValid();
}

inline
vector<float> LocallyConnected1D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs) const
{
    neuronNumber = neuronNumber % this->getNumberOfNeurons() / this->numberOfFilters;
    const int beginIndex = neuronNumber * this->shapeOfInput[1] * this->sizeOfFilterMatrix;
    const int endIndex = beginIndex + this->sizeOfFilterMatrix * this->shapeOfInput[1];

    if(endIndex <= this->shapeOfInput[1])
        return vector<float>(inputs.begin() + beginIndex, inputs.begin() + endIndex);
    else
    {
        auto v = vector<float>(inputs.begin() + beginIndex, inputs.begin() + this->shapeOfInput[1]);
        v.resize(this->sizeOfFilterMatrix, 0);
        return v;
    }
}

inline
void LocallyConnected1D::insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error,
                                                   std::vector<float>& errors) const
{
    neuronNumber = neuronNumber % this->getNumberOfNeurons() / this->numberOfFilters;
    const int beginIndex = neuronNumber * this->shapeOfInput[1] * this->sizeOfFilterMatrix;
    for (int e = 0; e < error.size(); ++e)
    {
        const int i = beginIndex + e;
        errors[i] += error[e];
    }
}

inline
bool LocallyConnected1D::operator==(const LocallyConnected1D& layer) const
{
    return this->Filter::operator==(layer);
}

inline
bool LocallyConnected1D::operator!=(const LocallyConnected1D& layer) const
{
    return !(*this == layer);
}
