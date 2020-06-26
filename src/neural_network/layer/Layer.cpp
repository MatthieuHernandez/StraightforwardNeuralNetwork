#include <boost/serialization/export.hpp>
#include "Layer.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Layer)

Layer::Layer(LayerModel& model, StochasticGradientDescent* optimizer)
{
    this->numberOfInputs = model.numberOfInputs;
    this->neurons.reserve(model.numberOfNeurons);
    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(model.numberOfInputsByNeurons, model.activation, optimizer);
    }
}

void Layer::train(vector<float>& inputErrors)
{
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        neurons[n].train(inputErrors[n]);
    }
}

int Layer::isValid() const
{
    if (this->neurons.size() != this->getNumberOfNeurons()
        || this->getNumberOfNeurons() < 1
        || this->getNumberOfNeurons() > 1000000)
        return 201;

    int numberOfOutput = 1;
    auto shape = this->getShapeOfOutput();
    for (int s : shape)
        numberOfOutput *= s;

    if (numberOfOutput != this->getNumberOfNeurons())
        return 202;

    for (auto& neuron : this->neurons)
    {
        int err = neuron.isValid();
        if (err != 0)
            return err;
    }
    return 0;
}

int Layer::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

int Layer::getNumberOfNeurons() const
{
    return this->neurons.size();
}

int Layer::getNumberOfParameters() const
{
    int sum = 0;
    for (auto& neuron : this->neurons)
    {
        sum += neuron.getNumberOfParameters();
    }
    return sum;
}

bool Layer::operator==(const Layer& layer) const
{
    return this->numberOfInputs == layer.numberOfInputs
        && this->errors == layer.errors
        && this->neurons == layer.neurons;
}

bool Layer::operator!=(const Layer& layer) const
{
    return !(*this == layer);
}
