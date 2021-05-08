#include "LayerOptimizerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "Dropout.hpp"
#include "L1Regularization.hpp"
#include "L2Regularization.hpp"
#include "ErrorMultiplier.hpp"

using namespace std;
using namespace snn;
using namespace internal;


LayerOptimizerModel snn::Dropout(float value)
{
    const LayerOptimizerModel model
    {
        layerOptimizerType::dropout,
        value
    };
    return model;
}

LayerOptimizerModel snn::L1Regularization(float value)
{
    const LayerOptimizerModel model
    {
        layerOptimizerType::l1Regularization,
        value
    };
    return model;
}

LayerOptimizerModel snn::L2Regularization(float value)
{
    const LayerOptimizerModel model
    {
        layerOptimizerType::l2Regularization,
        value
    };
    return model;
}

LayerOptimizerModel snn::ErrorMultiplier(float factor)
{
    const LayerOptimizerModel model
    {
        layerOptimizerType::errorMultiplier,
        factor
    };
    return model;
}


std::unique_ptr<LayerOptimizer> LayerOptimizerFactory::build(LayerOptimizerModel& model, BaseLayer* layer)
{
    switch (model.type)
    {
    case layerOptimizerType::dropout:
        return make_unique<Dropout>(model.value, layer);

    case layerOptimizerType::l1Regularization:
        return make_unique<L1Regularization>(model.value, layer);

    case layerOptimizerType::l2Regularization:
        return make_unique<L2Regularization>(model.value, layer);

    case layerOptimizerType::errorMultiplier:
        return make_unique<ErrorMultiplier>(model.value, layer);

    default:
        throw InvalidArchitectureException("Layer optimizer type is not implemented.");
    }
}

void LayerOptimizerFactory::build(vector<unique_ptr<LayerOptimizer>>& optimizers, LayerModel& model, BaseLayer* layer)
{
    for (auto& optimizer : model.optimizers)
    {
        optimizers.push_back(build(optimizer, layer));
    }
}
