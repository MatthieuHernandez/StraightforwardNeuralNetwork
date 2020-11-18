#pragma once
#include <memory>
#include <vector>
#include "LayerOptimizer.hpp"
#include "LayerOptimizerModel.hpp"
#include "../layer/LayerModel.hpp"

namespace snn
{
    extern LayerOptimizerModel Dropout(float value);

    namespace internal
    {
        class LayerOptimizerFactory
        {
        private:
            static std::unique_ptr<LayerOptimizer> build(LayerOptimizerModel& model);

        public:
            static void build(std::vector<std::unique_ptr<LayerOptimizer>>& optimizers, LayerModel& model);
        };
    }
}
