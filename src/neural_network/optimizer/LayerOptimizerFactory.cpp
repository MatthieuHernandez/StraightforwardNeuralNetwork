#include "LayerOptimizerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "Dropout.hpp"

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


std::unique_ptr<LayerOptimizer> LayerOptimizerFactory::build(LayerOptimizerModel& model)
{
    switch (model.type)
    {
    case layerOptimizerType::dropout:
        return make_unique<Dropout>(model.value);
    default:
        throw InvalidArchitectureException("Layer optimizer type is not implemented.");
    }
}

void LayerOptimizerFactory::build(vector<unique_ptr<LayerOptimizer>>& optimizers, LayerModel& model)
{
    for (auto& optimizer : model.optimizers)
    {
        optimizers.push_back(build(optimizer));
    }
}
