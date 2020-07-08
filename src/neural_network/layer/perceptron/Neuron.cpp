#include "Neuron.hpp"
#include "../../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Neuron::Neuron(const int numberOfInputs,
                       activationFunction activation,
                       StochasticGradientDescent* optimizer)
    : activation(activation),
      optimizer(optimizer)
{
    this->previousDeltaWeights.resize(numberOfInputs, 0);
    this->lastInputs.resize(numberOfInputs, 0);
    this->errors.resize(numberOfInputs, 0);
    this->outputFunction = ActivationFunction::get(this->activation);
    this->weights.resize(numberOfInputs);
    for (auto& w : weights)
    {
        w = randomInitializeWeight(numberOfInputs);
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
    return this->weights.size();
}

bool Neuron::operator==(const Neuron& Neuron) const
{
    return this->weights == Neuron.weights
        && this->bias == Neuron.bias
        && this->previousDeltaWeights == Neuron.previousDeltaWeights
        && this->lastInputs == Neuron.lastInputs
        && this->errors == Neuron.errors
        && this->lastOutput == Neuron.lastOutput
        && this->activation == Neuron.activation
        && this->outputFunction == Neuron.outputFunction // not really good
        && *this->optimizer == *Neuron.optimizer;
}

bool Neuron::operator!=(const Neuron& Neuron) const
{
    return !(*this == Neuron);
}
