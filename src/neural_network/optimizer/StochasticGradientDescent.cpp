#include "StochasticGradientDescent.hpp"

#include <boost/serialization/export.hpp>

#include "RecurrentNeuron.hpp"
#include "SimpleNeuron.hpp"

namespace snn::internal
{
StochasticGradientDescent::StochasticGradientDescent(const float learningRate, const float momentum)
    : learningRate(learningRate),
      momentum(momentum)
{
}

auto StochasticGradientDescent::clone() const -> std::shared_ptr<NeuralNetworkOptimizer>
{
    return std::make_shared<StochasticGradientDescent>(*this);
}
// #ifdef _MSC_VER
// #pragma warning(disable : 4701)
// #endif
void StochasticGradientDescent::updateWeights(SimpleNeuron& neuron, const float error) const
{
    auto w = 0;
    const auto& m = this->momentum;
    const auto& numberOfInputs = neuron.numberOfInputs;
    const auto& lastInputs = neuron.lastInputs.getSum();
    const auto& previousDeltaWeights = neuron.previousDeltaWeights;
    std::vector<float> deltaWeights(neuron.weights.size());
    const auto lr_error = this->learningRate * error;  // To activate the SIMD optimization.
    // #pragma omp simd
    for (w = 0; w < numberOfInputs; ++w)
    {
        deltaWeights[w] = lr_error * lastInputs[w] + m * previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights[w];
    }
    deltaWeights[w] = lr_error * neuron.bias + m * previousDeltaWeights[w];
    neuron.weights[w] += deltaWeights[w];
    neuron.previousDeltaWeights = deltaWeights;
}

void StochasticGradientDescent::updateWeights(RecurrentNeuron& neuron, float error) const
{
    auto w = 0;
    const auto& m = this->momentum;
    const auto& numberOfInputs = neuron.numberOfInputs;
    const auto& lastInputs = *neuron.lastInputs.getBack();
    const auto& previousDeltaWeights = neuron.previousDeltaWeights;
    std::vector<float> deltaWeights(neuron.weights.size());
    const auto lr_error = this->learningRate * error;
    // #pragma omp simd  // info C5002: Omp simd loop not vectorized due to reason '1305' (Not enough type information.)
    for (w = 0; w < numberOfInputs; ++w)
    {
        deltaWeights[w] = lr_error * lastInputs[w] + m * previousDeltaWeights[w];
        neuron.weights[w] += deltaWeights[w];
    }
    deltaWeights[w] = lr_error * neuron.bias + m * previousDeltaWeights[w];
    neuron.weights[w] += deltaWeights[w];
    neuron.recurrentError =
        error + neuron.recurrentError * neuron.outputFunction->derivative(neuron.previousSum) * neuron.weights[w];

    deltaWeights[w] = this->learningRate * neuron.recurrentError * neuron.previousOutput + m * previousDeltaWeights[w];
    neuron.weights[w] += deltaWeights[w];
    neuron.previousDeltaWeights = deltaWeights;
}
// #ifdef _MSC_VER
// #pragma warning(default : 4701)
// #endif

auto StochasticGradientDescent::isValid() const -> errorType
{
    if (this->learningRate <= 0.0F || this->learningRate >= 1.0F)
    {
        return errorType::optimizerWrongLearningRate;
    }
    if (this->momentum < 0.0F || this->momentum > 1.0F)
    {
        return errorType::optimizerWrongMomentum;
    }
    return errorType::noError;
}

auto StochasticGradientDescent::summary() const -> std::string
{
    std::stringstream summary;
    summary << " StochasticGradientDescent\n";
    summary << "                Learning rate: " << this->learningRate << '\n';
    summary << "                Momentum:      " << this->momentum << '\n';
    return summary.str();
}

auto StochasticGradientDescent::operator==(const NeuralNetworkOptimizer& optimizer) const -> bool
{
    try
    {
        const auto& o = dynamic_cast<const StochasticGradientDescent&>(optimizer);
        return this->NeuralNetworkOptimizer::operator==(optimizer) && this->learningRate == o.learningRate &&
               this->momentum == o.momentum;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

auto StochasticGradientDescent::operator!=(const NeuralNetworkOptimizer& optimizer) const -> bool
{
    return !(*this == optimizer);
}
}  // namespace snn::internal
