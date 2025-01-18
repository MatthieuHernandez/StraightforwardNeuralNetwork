#include "NeuralNetworkOptimizerFactory.hpp"

#include "Dropout.hpp"
#include "ExtendedExpection.hpp"
#include "StochasticGradientDescent.hpp"

using namespace std;
using namespace snn;
using namespace internal;

NeuralNetworkOptimizerModel snn::StochasticGradientDescent(float learningRate, float momentum)
{
    const NeuralNetworkOptimizerModel model{neuralNetworkOptimizerType::stochasticGradientDescent, learningRate,
                                            momentum};
    return model;
}

shared_ptr<NeuralNetworkOptimizer> NeuralNetworkOptimizerFactory::build(const NeuralNetworkOptimizerModel& model)
{
    switch (model.type)
    {
        case neuralNetworkOptimizerType::stochasticGradientDescent:
            return make_shared<StochasticGradientDescent>(model.learningRate, model.momentum);
        default:
            throw InvalidArchitectureException("Neural network optimizer type is not implemented.");
    }
}
