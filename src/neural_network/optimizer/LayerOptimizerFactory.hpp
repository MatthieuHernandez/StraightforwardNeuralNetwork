#pragma once
#include <memory>
#include <vector>
#include "LayerOptimizer.hpp"
#include "OptimizerModel.hpp"
#include "../layer/LayerModel.hpp"

namespace snn
{
    extern OptimizerModel Dropout(float value);

    namespace internal
    {
        class LayerOptimizerFactory
        {
        private:
            static std::unique_ptr<LayerOptimizer> build(OptimizerModel& model, LayerModel& layerModel);

        public:
            static void build(std::vector<std::unique_ptr<LayerOptimizer>>& optimizers, std::vector<OptimizerModel>& models, LayerModel& model);
        };
    }
}
