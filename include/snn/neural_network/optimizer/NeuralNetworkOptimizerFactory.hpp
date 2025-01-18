#pragma once
#include <memory>

#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkOptimizerModel.hpp"

namespace snn
{
extern NeuralNetworkOptimizerModel StochasticGradientDescent(float learningRate = 0.03f, float momentum = 0.0f);

namespace internal
{
class NeuralNetworkOptimizerFactory
{
    public:
        static std::shared_ptr<NeuralNetworkOptimizer> build(const NeuralNetworkOptimizerModel& model);
};
}  // namespace internal
}  // namespace snn
