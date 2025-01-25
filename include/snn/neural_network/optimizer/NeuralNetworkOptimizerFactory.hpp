#pragma once
#include <memory>

#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkOptimizerModel.hpp"

namespace snn
{
extern auto StochasticGradientDescent(float learningRate = 0.03f, float momentum = 0.0f) -> NeuralNetworkOptimizerModel;

namespace internal
{
class NeuralNetworkOptimizerFactory
{
    public:
        static auto build(const NeuralNetworkOptimizerModel& model) -> std::shared_ptr<NeuralNetworkOptimizer>;
};
}  // namespace internal
}  // namespace snn
