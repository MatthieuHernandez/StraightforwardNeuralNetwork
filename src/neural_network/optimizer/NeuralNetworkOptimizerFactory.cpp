#include "NeuralNetworkOptimizerFactory.hpp"

#include "ExtendedExpection.hpp"
#include "StochasticGradientDescent.hpp"

namespace snn
{
auto StochasticGradientDescent(float learningRate, float momentum) -> NeuralNetworkOptimizerModel
{
    const NeuralNetworkOptimizerModel model{.type = neuralNetworkOptimizerType::stochasticGradientDescent,
                                            .learningRate = learningRate,
                                            .momentum = momentum};
    return model;
}
namespace internal
{
auto NeuralNetworkOptimizerFactory::build(const NeuralNetworkOptimizerModel& model)
    -> std::shared_ptr<NeuralNetworkOptimizer>
{
    switch (model.type)
    {
        case neuralNetworkOptimizerType::stochasticGradientDescent:
            return std::make_shared<StochasticGradientDescent>(model.learningRate, model.momentum);
        default:
            throw InvalidArchitectureException("Neural network optimizer type is not implemented.");
    }
}
}  // namespace internal
}  // namespace snn