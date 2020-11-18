#include "NeuralNetworkOptimizerFactory.hpp"
#include "StochasticGradientDescent.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "Dropout.hpp"

using namespace std;
using namespace snn;
using namespace internal;


NeuralNetworkOptimizerModel StochasticGradientDescent(float value)
{
    const NeuralNetworkOptimizerModel model
    {
        neuralNetworkOptimizerType::stochasticGradientDescent,
        value
    };
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
