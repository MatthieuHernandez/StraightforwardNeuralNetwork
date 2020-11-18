#pragma once
#include <memory>
#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkOptimizerModel.hpp"

namespace snn
{
    extern NeuralNetworkOptimizerModel StochasticGradientDescent(float value);

    namespace internal
    {
        class NeuralNetworkOptimizerFactory
        {
        public:
            static std::shared_ptr<NeuralNetworkOptimizer> build(const NeuralNetworkOptimizerModel& model);
        };
    }
}
