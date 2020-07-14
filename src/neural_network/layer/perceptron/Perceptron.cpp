#include <boost/serialization/export.hpp>
#include "Perceptron.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;


Perceptron::Perceptron(int numberOfInputs, activationFunction activation, StochasticGradientDescent* optimizer)
    : Neuron(numberOfInputs, activation, optimizer)
{
}

int Perceptron::isValid() const
{
    return this->Neuron::isValid();
}

bool Perceptron::operator==(const Neuron& neuron) const
{
    return this->Neuron::operator==(neuron);
}

bool Perceptron::operator!=(const Neuron& neuron) const
{
    return !(*this == neuron);
}
