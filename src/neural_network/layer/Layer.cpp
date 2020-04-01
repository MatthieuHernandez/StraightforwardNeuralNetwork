#include <boost/serialization/export.hpp>
#include "Layer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Layer)

Layer::Layer(layerType type,
             int numberOfInputs,
             int numberOfNeurons)
{
    this->numberOfInputs = numberOfInputs;
    this->neurons.reserve(numberOfNeurons);
}


vector<float> Layer::output(const vector<float>& inputs)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = this->createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(neuronInputs);
    }
    return outputs;
}

vector<float> Layer::backOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = neurons[n].backOutput(inputErrors[n]);
         this->insertBackOutputForNeuron(n, error, errors);
    }
    return errors;
}


void Layer::train(vector<float>& inputErrors)
{
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        neurons[n].backOutput(inputErrors[n]);
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

    if(numberOfOutput != this->getNumberOfNeurons())
        return 202;

    for (auto& neuron : this->neurons)
    {
        int err = neuron.isValid();
        if(err != 0)
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
