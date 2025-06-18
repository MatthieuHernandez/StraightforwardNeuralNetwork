#include "StochasticGradientDescent.hpp"

#include <boost/serialization/export.hpp>
#include <sstream>

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
void StochasticGradientDescent::updateWeights(SimpleNeuron& neuron) const
{
    auto w = 0;
    const auto& m = this->momentum;
    const auto& numberOfInputs = neuron.numberOfInputs;
    const auto error = neuron.lastError.getSum();
    const auto input_error = neuron.lastInputs.MultiplyAndAccumulate(neuron.lastError);
    auto& deltaWeights = neuron.deltaWeights;
    auto& weights = neuron.weights;
    const auto lr = this->learningRate;
    // #pragma omp simd
    for (w = 0; w < numberOfInputs; ++w)
    {
        deltaWeights[w] = lr * input_error[w] + m * deltaWeights[w];
        weights[w] += deltaWeights[w];
    }
    deltaWeights[w] = lr * error * neuron.bias + m * deltaWeights[w];
    weights[w] += deltaWeights[w];
}

void StochasticGradientDescent::updateWeights(RecurrentNeuron& neuron) const
{
    auto w = 0;
    const auto& m = this->momentum;
    const auto& numberOfInputs = neuron.numberOfInputs;
    const auto error = neuron.lastError.getSum();
    const auto input_error = neuron.lastInputs.MultiplyAndAccumulate(neuron.lastError);
    auto& deltaWeights = neuron.deltaWeights;
    auto& weights = neuron.weights;
    const auto lr = this->learningRate;
    // #pragma omp simd  // info C5002: Omp simd loop not vectorized due to reason '1305' (Not enough type information.)
    for (w = 0; w < numberOfInputs; ++w)
    {
        deltaWeights[w] = lr * input_error[w] + m * deltaWeights[w];
        weights[w] += deltaWeights[w];
    }
    // TODO(matth): previousOutput should be a Circular like lastInputs and do previousOutput.MultiplyAndAccumulate
    // (neuron.lastError). And also rename previousOutput as lastOutput.
    deltaWeights[w] = this->learningRate * neuron.recurrentError * neuron.previousOutput + m * neuron.deltaWeights[w];
    weights[w] += deltaWeights[w];
    neuron.recurrentError = error;  // + neuron.recurrentError *
                                    // neuron.outputFunction->derivative(neuron.previousSum) * weights[w];

    w++;
    deltaWeights[w] = lr * error * neuron.bias + m * deltaWeights[w];
    weights[w] += deltaWeights[w];
}
// #ifdef _MSC_VER
// #pragma warning(default : 4701)
// #endif

auto StochasticGradientDescent::isValid() const -> errorType
{
    if (this->learningRate < 0.0F || this->learningRate >= 1.0F)
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
