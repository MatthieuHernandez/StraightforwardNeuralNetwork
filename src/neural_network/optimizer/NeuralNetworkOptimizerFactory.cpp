#include "NeuralNetworkOptimizerFactory.hpp"
#include "Adam.hpp"
#include "StochasticGradientDescent.hpp"
#include "../../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;


NeuralNetworkOptimizerModel snn::StochasticGradientDescent(const float learningRate, const float momentum)
{
    const NeuralNetworkOptimizerModel model
    {
        neuralNetworkOptimizerType::stochasticGradientDescent,
        learningRate,
        momentum
    };
    return model;
}

NeuralNetworkOptimizerModel snn::Adam(const float learningRate, const float beta1, const float beta2, const float epsilon)
{
    const NeuralNetworkOptimizerModel model
    {
        neuralNetworkOptimizerType::adam,
        learningRate,
        0.0f,
        beta1,
        beta2,
        epsilon
    };
    return model;
}

shared_ptr<NeuralNetworkOptimizer> NeuralNetworkOptimizerFactory::build(const NeuralNetworkOptimizerModel& model)
{
    switch (model.type)
    {
    case neuralNetworkOptimizerType::stochasticGradientDescent:
        return make_shared<StochasticGradientDescent>(model.learningRate, model.momentum);
    case neuralNetworkOptimizerType::adam:
        return make_shared<Adam>(model.learningRate, model.beta1, model.beta2, model.epsilon);
    default:
        throw InvalidArchitectureException("Neural network optimizer type is not implemented.");
    }
}
