#include <cmath>
#include "NeuralNetwork.hpp"
#include "../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

vector<float> NeuralNetwork::output(const vector<float>& inputs, bool temporalReset)
{
    auto outputs = layers[0]->output(inputs, temporalReset);

    for (int l = 1; l < this->layers.size(); ++l)
    {
        outputs = layers[l]->output(outputs, temporalReset);
    }

    return outputs;
}

void NeuralNetwork::evaluateOnceForRegression(
    const vector<float>& inputs, const vector<float>& desired, const float precision, bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForRegression(outputs, desired, precision);
}

void NeuralNetwork::evaluateOnceForMultipleClassification(
    const vector<float>& inputs, const vector<float>& desired, const float separator, bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForMultipleClassification(outputs, desired, separator);
}

void NeuralNetwork::evaluateOnceForClassification(const vector<float>& inputs, const int classNumber,
                                                  bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForClassification(outputs, classNumber);
}

void NeuralNetwork::trainOnce(const vector<float>& inputs, const vector<float>& desired, bool temporalReset)
{
    this->backpropagationAlgorithm(inputs, desired, temporalReset);
}

void NeuralNetwork::backpropagationAlgorithm(const vector<float>& inputs, const vector<float>& desired,
                                             bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    auto errors = calculateError(outputs, desired);

    for (int l = this->layers.size() - 1; l > 0; --l)
    {
        errors = layers[l]->backOutput(errors);
    }
    layers[0]->train(errors);
}

inline
vector<float>& NeuralNetwork::calculateError(const vector<float>& outputs, const vector<float>& desired) const
{
    auto errors = new vector<float>(this->layers.back()->getNumberOfNeurons(), 0);
    for (int n = 0; n < errors->size(); ++n)
    {
        if (isnan(desired[n]))
            (*errors)[n] = 0;
        else
        {
            (*errors)[n] = 2 * (desired[n] - outputs[n]);
        }
    }
    return *errors;
}
