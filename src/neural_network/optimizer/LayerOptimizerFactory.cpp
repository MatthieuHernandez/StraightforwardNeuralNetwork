#include "LayerOptimizerFactory.hpp"

#include "Dropout.hpp"
#include "ErrorMultiplier.hpp"
#include "ExtendedExpection.hpp"
#include "L1Regularization.hpp"
#include "L2Regularization.hpp"
#include "Softmax.hpp"

namespace snn
{
auto Dropout(float value) -> LayerOptimizerModel
{
    const LayerOptimizerModel model{layerOptimizerType::dropout, value};
    return model;
}

auto L1Regularization(float value) -> LayerOptimizerModel
{
    const LayerOptimizerModel model{layerOptimizerType::l1Regularization, value};
    return model;
}

auto L2Regularization(float value) -> LayerOptimizerModel
{
    const LayerOptimizerModel model{layerOptimizerType::l2Regularization, value};
    return model;
}

auto ErrorMultiplier(float factor) -> LayerOptimizerModel
{
    const LayerOptimizerModel model{layerOptimizerType::errorMultiplier, factor};
    return model;
}

auto Softmax() -> LayerOptimizerModel
{
    const LayerOptimizerModel model{layerOptimizerType::softmax, 0.0};
    return model;
}
namespace internal
{
auto LayerOptimizerFactory::build(LayerOptimizerModel& model, BaseLayer* layer) -> std::unique_ptr<LayerOptimizer>
{
    switch (model.type)
    {
        case layerOptimizerType::dropout:
            return std::make_unique<Dropout>(model.value, layer);

        case layerOptimizerType::l1Regularization:
            return std::make_unique<L1Regularization>(model.value, layer);

        case layerOptimizerType::l2Regularization:
            return std::make_unique<L2Regularization>(model.value, layer);

        case layerOptimizerType::errorMultiplier:
            return std::make_unique<ErrorMultiplier>(model.value, layer);

        case layerOptimizerType::softmax:
            return std::make_unique<Softmax>(layer);

        default:
            throw InvalidArchitectureException("Layer optimizer type is not implemented.");
    }
}

void LayerOptimizerFactory::build(std::vector<std::unique_ptr<LayerOptimizer>>& optimizers, LayerModel& model,
                                  BaseLayer* layer)
{
    for (auto& optimizer : model.optimizers)
    {
        optimizers.push_back(build(optimizer, layer));
    }
}
}  // namespace internal
}  // namespace snn