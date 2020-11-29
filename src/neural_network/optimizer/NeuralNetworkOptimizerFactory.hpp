#pragma once
#include <memory>
#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkOptimizerModel.hpp"

namespace snn
{
    extern NeuralNetworkOptimizerModel StochasticGradientDescent(float learningRate  = 0.03f, float momentum = 0.0f);
    extern NeuralNetworkOptimizerModel Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    namespace internal
    {
        class NeuralNetworkOptimizerFactory
        {
        public:
            static std::shared_ptr<NeuralNetworkOptimizer> build(const NeuralNetworkOptimizerModel& model);
        };
    }
}

