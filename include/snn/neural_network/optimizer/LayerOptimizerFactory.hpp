#pragma once
#include <memory>
#include <vector>

#include "../layer/LayerModel.hpp"
#include "LayerOptimizer.hpp"
#include "LayerOptimizerModel.hpp"

namespace snn
{
extern auto Dropout(float value) -> LayerOptimizerModel;
extern auto L1Regularization(float value) -> LayerOptimizerModel;
extern auto L2Regularization(float value) -> LayerOptimizerModel;
extern auto ErrorMultiplier(float factor) -> LayerOptimizerModel;
extern auto Softmax() -> LayerOptimizerModel;

namespace internal
{
class LayerOptimizerFactory
{
    private:
        static auto build(LayerOptimizerModel& model, BaseLayer* layer) -> std::unique_ptr<LayerOptimizer>;

    public:
        static void build(std::vector<std::unique_ptr<LayerOptimizer>>& optimizers, LayerModel& model,
                          BaseLayer* layer);
};
}  // namespace internal
}  // namespace snn
