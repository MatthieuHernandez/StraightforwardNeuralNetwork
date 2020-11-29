#include <boost/serialization/export.hpp>
#include "Adam.hpp"
#include "../layer/neuron/RecurrentNeuron.hpp"
#include "../layer/neuron/SimpleNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Adam)

Adam::Adam(const float learningRate = 0.001f, const float beta1 = 0.9f, const float beta2 = 0.999f, const float epsilon = 1e-8f)
    : learningRate(learningRate),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon)
{
}

shared_ptr<NeuralNetworkOptimizer> Adam::clone() const
{
    return make_shared<Adam>(*this);
}

void Adam::updateWeights(SimpleNeuron& neuron, float error) const
{
    for (size_t w = 0; w < neuron.weights.size(); ++w)
    {
        auto delta = error * neuron.lastInputs[w];
        auto m = this->beta1 * neuron.firstMomentWeights[w] + (1- this->beta1) * delta;
        auto v = this->beta2 * neuron.secondRawMomentWeights[w] + (1 - this->beta2) * sqrtf(delta);
        auto correctedM = m / (1 - powf(m, this->t));
        auto correctedV = v / (1 -  powf(m, this->t));
        auto deltaWeights = this->learningRate * correctedM / (sqrtf(correctedV) + this->epsilon);
        neuron.weights[w] += deltaWeights;
        neuron.firstMomentWeights[w] = m;
        neuron.secondRawMomentWeights[w] = v;
    }
}

void Adam::updateWeights(RecurrentNeuron& neuron, float error) const
{
    size_t w;
    for (w = 0; w < neuron.weights.size(); ++w)
    {
        auto delta = error * neuron.lastInputs[w];
        auto m = this->beta1 * neuron.firstMomentWeights[w] + (1- this->beta1) * delta;
        auto v = this->beta2 * neuron.secondRawMomentWeights[w] + (1 - this->beta2) * sqrtf(delta);
        auto correctedM = m / (1 - powf(m, this->t));
        auto correctedV = v / (1 -  powf(m, this->t));
        auto deltaWeights = this->learningRate * correctedM / (sqrtf(correctedV) + this->epsilon);
        neuron.weights[w] += deltaWeights;
        neuron.firstMomentWeights[w] = m;
        neuron.secondRawMomentWeights[w] = v;
    }
    neuron.recurrentError = error + neuron.recurrentError * neuron.outputFunction->derivative(neuron.previousSum) * neuron.weights[w];

    auto delta = neuron.recurrentError * neuron.lastInputs[w];
    auto m = this->beta1 * neuron.firstMomentWeights[w] + (1 - this->beta1) * delta;
    auto v = this->beta2 * neuron.secondRawMomentWeights[w] + (1 - this->beta2) * sqrtf(delta);
    auto correctedM = m / (1 - powf(m, this->t));
    auto correctedV = v / (1 - powf(m, this->t));
    auto deltaWeights = this->learningRate * correctedM / (sqrtf(correctedV) + this->epsilon);
    neuron.weights[w] += deltaWeights;
    neuron.firstMomentWeights[w] = m;
    neuron.secondRawMomentWeights[w] = v;
}

int Adam::isValid()
{
    if (this->learningRate <= 0.0f || this->learningRate >= 1.0f)
        return 501;
    if (this->beta1 < 0.0f || this->beta1 > 1.0f)
        return 503;
    if (this->beta2 < 0.0f || this->beta2 > 1.0f)
        return 504;
    if (this->epsilon < 0.0f || this->epsilon > 1.0f)
        return 505;
    return 0;
}

bool Adam::operator==(const NeuralNetworkOptimizer& optimizer) const
{
    try
    {
        const auto& o = dynamic_cast<const Adam&>(optimizer);
        return this->NeuralNetworkOptimizer::operator==(optimizer)
            && this->learningRate == o.learningRate
            && this->beta1 == o.beta1
            && this->beta2 == o.beta2
            && this->epsilon == o.epsilon;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool Adam::operator!=(const NeuralNetworkOptimizer& optimizer) const
{
    return !(*this == optimizer);
}
