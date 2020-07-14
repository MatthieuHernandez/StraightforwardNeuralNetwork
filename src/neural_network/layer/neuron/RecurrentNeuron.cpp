#include <boost/serialization/export.hpp>
#include "RecurrentNeuron.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

BOOST_CLASS_EXPORT(Neuron)

RecurrentNeuron::RecurrentNeuron(NeuronModel model, StochasticGradientDescent* optimizer)
    : Neuron(model, optimizer)
{
    previousOutputs.resize(model.numberOfRecurrences);
}

int RecurrentNeuron::isValid() const
{
    return this->Neuron::isValid();
}

bool RecurrentNeuron::operator==(const Neuron& neuron) const
{
    return this->Neuron::operator==(neuron);
}

bool RecurrentNeuron::operator!=(const Neuron& neuron) const
{
    return !(*this == neuron);
}