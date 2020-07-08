#include <cmath>
#include "Perceptron.hpp"
#include "../../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;


int Perceptron::isValid() const
{
    return this->Neuron::isValid();
}

bool Perceptron::operator==(const Perceptron& perceptron) const
{
    return this->Neuron::operator==(perceptron);
}

bool Perceptron::operator!=(const Perceptron& perceptron) const
{
    return !(*this == perceptron);
}
