#include "StochasticGradientDescent.hpp"

#include <boost/serialization/export.hpp>
#include <sstream>

#include "Neuron.hpp"

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

void StochasticGradientDescent::updateWeights(Neuron& neuron) const
{
    const auto& m = this->momentum;
    const auto input_error = neuron.lastInputs.MultiplyAndAccumulate(neuron.lastError);
    auto& deltaWeights = neuron.deltaWeights;
    auto& weights = neuron.weights;
    const auto lr = this->learningRate;
    for (size_t w = 0; w < neuron.weights.size(); ++w)
    {
        deltaWeights[w] = (lr * input_error[w]) + (m * deltaWeights[w]);
        weights[w] += deltaWeights[w];
    }
}

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
