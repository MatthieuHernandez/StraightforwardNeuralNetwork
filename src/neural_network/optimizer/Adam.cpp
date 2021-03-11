#include <boost/serialization/export.hpp>
#include "Adam.hpp"
#include "../layer/neuron/RecurrentNeuron.hpp"
#include "../layer/neuron/SimpleNeuron.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Adam)

Adam::Adam(const float learningRate, const float beta1, const float beta2, const float epsilon)
    : learningRate(learningRate),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon)
{
    this->reverseBeta1 = 1.0f - this->beta1;
    this->reverseBeta2 = 1.0f - this->beta2;
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
        neuron.firstMomentWeights[w] = this->beta1 * neuron.firstMomentWeights[w] + this->reverseBeta1 * delta;
        neuron.secondRawMomentWeights[w] = this->beta2 * neuron.secondRawMomentWeights[w] + this->reverseBeta2 * delta * delta;
        //auto correctedM = neuron.firstMomentWeights[w] / this->precomputedM;
        //auto correctedV = neuron.secondRawMomentWeights[w] / this->precomputedV;
        auto deltaWeights = this->precomputedDelta * neuron.firstMomentWeights[w] / (sqrtf(neuron.secondRawMomentWeights[w]) + this->epsilon);
        neuron.weights[w] += deltaWeights;
    }
}

void Adam::updateWeights(RecurrentNeuron& neuron, float error) const
{
    size_t w;
    for (w = 0; w < neuron.weights.size(); ++w)
    {
        auto delta = error * neuron.lastInputs[w];
        neuron.firstMomentWeights[w] = this->beta1 * neuron.firstMomentWeights[w] + this->reverseBeta1 * delta;
        neuron.secondRawMomentWeights[w] = this->beta2 * neuron.secondRawMomentWeights[w] + this->reverseBeta2 * delta * delta;
        auto correctedM = neuron.firstMomentWeights[w] / this->precomputedM;
        auto correctedV = neuron.secondRawMomentWeights[w] / this->precomputedV;
        auto deltaWeights = this->learningRate * correctedM / (sqrtf(correctedV) + this->epsilon);
        neuron.weights[w] += deltaWeights;
    }
    neuron.recurrentError = error + neuron.recurrentError * neuron.outputFunction->derivative(neuron.previousSum) * neuron.weights[w];

    auto delta = neuron.recurrentError * neuron.lastInputs[w];
    neuron.firstMomentWeights[w] = this->beta1 * neuron.firstMomentWeights[w] + this->reverseBeta1 * delta;
    neuron.secondRawMomentWeights[w] = this->beta2 * neuron.secondRawMomentWeights[w] + this->reverseBeta2 * delta * delta;
    auto correctedM = neuron.firstMomentWeights[w] / this->precomputedM;
    auto correctedV = neuron.secondRawMomentWeights[w] / this->precomputedV;
    auto deltaWeights = this->learningRate * correctedM / (sqrtf(correctedV) + this->epsilon);
    neuron.weights[w] += deltaWeights;
}

void Adam::operator++()
{
    NeuralNetworkOptimizer::operator++();
    this->precomputedM = 1.0f - powf(this->beta1, this->t);
    this->precomputedV = 1.0f - powf(this->beta2, this->t);
    this->precomputedDelta = this->learningRate * sqrtf(this->precomputedV) / this->precomputedM;
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
