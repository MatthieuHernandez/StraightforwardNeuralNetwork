#include <boost/serialization/export.hpp>
#include "RecurrentNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Neuron)

RecurrentNeuron::RecurrentNeuron(int numberOfInputs, activationFunction activation, StochasticGradientDescent* optimizer)
    : Neuron(numberOfInputs, activation, optimizer)
{
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