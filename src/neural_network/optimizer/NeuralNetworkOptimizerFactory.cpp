#include "NeuralNetworkOptimizerFactory.hpp"

#include "Dropout.hpp"
#include "ExtendedExpection.hpp"
#include "StochasticGradientDescent.hpp"

using namespace std;
using namespace snn;
using namespace internal;

auto snn::StochasticGradientDescent(float learningRate, float momentum) -> NeuralNetworkOptimizerModel
{
    const NeuralNetworkOptimizerModel model{neuralNetworkOptimizerType::stochasticGradientDescent, learningRate,
                                            momentum};
    return model;
}

auto NeuralNetworkOptimizerFactory::build(const NeuralNetworkOptimizerModel& model)
    -> shared_ptr<NeuralNetworkOptimizer>
{
    switch (model.type)
    {
        case neuralNetworkOptimizerType::stochasticGradientDescent:
            return make_shared<StochasticGradientDescent>(model.learningRate, model.momentum);
        default:
            throw InvalidArchitectureException("Neural network optimizer type is not implemented.");
    }
}
