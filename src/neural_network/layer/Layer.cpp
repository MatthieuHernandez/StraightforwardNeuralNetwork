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
    this->numberOfNeurons = numberOfNeurons;
    this->neurons.reserve(numberOfNeurons);
}

int Layer::isValid() const
{
    if(this->neurons.size() != this->numberOfNeurons
        || this->numberOfNeurons < 1
        || this->numberOfNeurons > 1000000)
        return 201;

    for (auto& neuron : this->neurons)
    {
        int err = neuron.isValid();
        if(err != 0)
            return err;
    }
    return 0;
}

bool Layer::operator==(const Layer& layer) const
{
    return this->numberOfInputs == layer.numberOfInputs
        && this->numberOfNeurons == layer.numberOfNeurons
        && this->errors == layer.errors
        && this->neurons == layer.neurons;
}

bool Layer::operator!=(const Layer& layer) const
{
    return !this->operator==(layer);
}
