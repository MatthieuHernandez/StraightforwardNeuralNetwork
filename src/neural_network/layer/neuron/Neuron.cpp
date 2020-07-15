#include <typeinfo>
#include "Neuron.hpp"
#include "../../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

Neuron::Neuron(NeuronModel model,
                       StochasticGradientDescent* optimizer)
    : activation(model.activation),
      optimizer(optimizer)
{
    this->previousDeltaWeights.resize(model.numberOfInputs, 0);
    this->lastInputs.resize(model.numberOfInputs, 0);
    this->errors.resize(model.numberOfInputs, 0);
    this->outputFunction = ActivationFunction::get(this->activation);
    this->weights.resize(model.numberOfInputs);
    for (auto& w : weights)
    {
        w = randomInitializeWeight(model.numberOfInputs);
    }
    this->bias = 1.0f;
}

float Neuron::randomInitializeWeight(int numberOfInputs) const
{
    const float valueMax = 2.4f / sqrtf(static_cast<float>(numberOfInputs));
    return Tools::randomBetween(-valueMax, valueMax);
}

float Neuron::output(const vector<float>& inputs)
{
    lastInputs = inputs;
    float sum = 0;
    for (int w = 0; w < this->weights.size(); ++w)
    {
        sum += inputs[w] * weights[w];
    }
    sum += bias;
    lastOutput = sum;
    sum = outputFunction->function(sum);
    return sum;
}

std::vector<float>& Neuron::backOutput(float error)
{
    error = error * outputFunction->derivative(lastOutput);

    this->updateWeights(lastInputs, error);

    for (int w = 0; w < this->weights.size(); ++w)
    {
        errors[w] = error * weights[w];
    }
    return errors;
}

void Neuron::train(float error)
{
    error = error * outputFunction->derivative(lastOutput);

    this->updateWeights(lastInputs, error);
}

void Neuron::updateWeights(const std::vector<float>& inputs, const float error)
{
    for (int w = 0; w < this->weights.size(); ++w)
    {
        auto deltaWeights = this->optimizer->learningRate * error * inputs[w];
        deltaWeights += this->optimizer->momentum * this->previousDeltaWeights[w];
        weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
}

int Neuron::isValid() const
{
    if (this->bias != 1.0f)
        return 301;

    if (this->weights.empty()
        || this->weights.size() > 1000000)
    {
        return 302;
    }
    for (auto& weight : weights)
        if (weight < -100000 || weight > 10000)
            return 303;

    return 0;
}

vector<float> Neuron::getWeights() const
{
    return this->weights;
}

int Neuron::getNumberOfParameters() const
{
    return static_cast<int>(this->weights.size());
}

void Neuron::setWeights(const vector<float>& weights)
{
    this->weights = weights;
}

float Neuron::getWeight(const int w) const
{
    return weights[w];
}

void Neuron::setWeight(const int w, const float weight)
{
    this->weights[w] = weight;
}

float Neuron::getBias() const
{
    return bias;
}

void Neuron::setBias(const float bias)
{
    this->bias = bias;
}

int Neuron::getNumberOfInputs() const
{
    return static_cast<int>(this->weights.size());
}

bool Neuron::operator==(const Neuron& neuron) const
{
    return typeid(*this).hash_code() == typeid(neuron).hash_code()
        && this->weights == neuron.weights
        && this->bias == neuron.bias
        && this->previousDeltaWeights == neuron.previousDeltaWeights
        && this->lastInputs == neuron.lastInputs
        && this->errors == neuron.errors
        && this->lastOutput == neuron.lastOutput
        && this->activation == neuron.activation
        && this->outputFunction == neuron.outputFunction // not really good
        && *this->optimizer == *neuron.optimizer;
}

bool Neuron::operator!=(const Neuron& Neuron) const
{
    return !(*this == Neuron);
}
