#pragma once
#include <memory>
#include <vector>
#include "LayerOptimizer.hpp"
#include "LayerOptimizerModel.hpp"
#include "../layer/LayerModel.hpp"

namespace snn
{
    extern LayerOptimizerModel Dropout(float value);
    extern LayerOptimizerModel L1Regularization(float value);
    extern LayerOptimizerModel L2Regularization(float value);
    extern LayerOptimizerModel ErrorMultiplier(float factor);

    namespace internal
    {
        class LayerOptimizerFactory
        {
        private:
            static std::unique_ptr<LayerOptimizer> build(LayerOptimizerModel& model, BaseLayer* layer);

        public:
            static void build(std::vector<std::unique_ptr<LayerOptimizer>>& optimizers, LayerModel& model, BaseLayer* layer);
        };
    }
}
