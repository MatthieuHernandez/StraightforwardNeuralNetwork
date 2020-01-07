#include <cmath>
#include "Perceptron.hpp"
#include "../../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Perceptron::Perceptron(const int numberOfInputs,
                       activationFunction activation,
                       float* learningRate,
                       float* momentum)
    : activation(activation),
      learningRate(learningRate),
      momentum(momentum)
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

float Perceptron::randomInitializeWeight(int numberOfInputs) const
{
    const float valueMax = 2.4f / sqrt(static_cast<float>(numberOfInputs));
    return Tools::randomBetween(-valueMax, valueMax);
}

float Perceptron::output(const vector<float>& inputs)
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

std::vector<float>& Perceptron::backOutput(float error)
{
    error = error * outputFunction->derivative(lastOutput);

    this->train(lastInputs, error);

    for (int w = 0; w < this->weights.size(); ++w)
    {
        errors[w] = error * weights[w];
    }
    return errors;
}

void Perceptron::train(const std::vector<float>& inputs, const float error)
{
    for (int w = 0; w < this->weights.size(); ++w)
    {
        auto deltaWeights = *this->learningRate * error * inputs[w];
        deltaWeights += *this->momentum * this->previousDeltaWeights[w];
        weights[w] += deltaWeights;
        this->previousDeltaWeights[w] = deltaWeights;
    }
}

int Perceptron::isValid() const
{
    if (this->bias != 1)
        return 301;

    if (this->weights.size() < 1
     || this->weights.size() > 1000000)
    {
        return 302;
    }
    for (auto& weight : weights)
        if (weight < -100000 || weight > 10000)
            return 303;

    return 0;
}

vector<float> Perceptron::getWeights() const
{
    return weights;
}

void Perceptron::setWeights(const vector<float>& weights)
{
    this->weights = weights;
}

float Perceptron::getWeight(const int w) const
{
    return weights[w];
}

void Perceptron::setWeight(const int w, const float weight)
{
    this->weights[w] = weight;
}

float Perceptron::getBias() const
{
    return bias;
}

void Perceptron::setBias(const float bias)
{
    this->bias = bias;
}

int Perceptron::getNumberOfInputs() const
{
    return this->weights.size();
}

bool Perceptron::operator==(const Perceptron& perceptron) const
{
    return this->weights == perceptron.weights
        && this->previousDeltaWeights == perceptron.previousDeltaWeights
        && this->lastInputs == perceptron.lastInputs
        && this->errors == perceptron.errors
        && this->lastOutput == perceptron.lastOutput
        && this->bias == perceptron.bias
        && this->activation == perceptron.activation
        && this->outputFunction == perceptron.outputFunction;
}

bool Perceptron::operator!=(const Perceptron& perceptron) const
{
    return !(*this == perceptron);
}
