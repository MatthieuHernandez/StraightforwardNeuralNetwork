#include <cmath>
#include <boost/serialization/export.hpp>
#include "GateRecurrentUnit.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(GateRecurrentUnit)

GateRecurrentUnit::GateRecurrentUnit(NeuronModel model, StochasticGradientDescent* optimizer)
    : Neuron(model, optimizer)
{
}

float GateRecurrentUnit::output(const vector<float>& inputs, bool temporalReset)
{
    return 0;
}

std::vector<float>& GateRecurrentUnit::backOutput(float error)
{
    error = error * outputFunction->derivative(this->sum);
    this->updateWeights(this->lastInputs, error);

    for (int w = 0; w < this->numberOfInputs; ++w)
    {
        this->errors[w] = error * this->weights[w];
    }
    return this->errors;
}

void GateRecurrentUnit::train(float error)
{
    error = error * outputFunction->derivative(this->sum);
    this->updateWeights(this->lastInputs, error);
}

inline
void GateRecurrentUnit::updateWeights(const std::vector<float>& inputs, float error)
{
    
}

inline
void GateRecurrentUnit::reset()
{
    this->previousOutput = 0;
    this->recurrentError = 0;
    this->previousSum = 0;
}

int GateRecurrentUnit::isValid() const
{
    if (static_cast<int>(this->weights.size()) != this->numberOfInputs + 1)
        return 304;
    return this->Neuron::isValid();
}

bool GateRecurrentUnit::operator==(const Neuron& neuron) const
{
    try
    {
        const auto& n = dynamic_cast<const GateRecurrentUnit&>(neuron);
        return this->Neuron::operator==(neuron)
            && this->lastOutput == n.lastOutput
            && this->previousOutput == n.previousOutput
            && this->recurrentError == n.recurrentError
            && this->previousSum == n.previousSum;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool GateRecurrentUnit::operator!=(const Neuron& neuron) const
{
    return !(*this == neuron);
}
