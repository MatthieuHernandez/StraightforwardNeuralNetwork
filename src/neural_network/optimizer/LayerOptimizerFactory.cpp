#include "LayerOptimizerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "Dropout.hpp"

using namespace std;
using namespace snn;
using namespace internal;


OptimizerModel snn::Dropout(float value)
{
    OptimizerModel model
    {
    };
    return model;
}


std::unique_ptr<LayerOptimizer> LayerOptimizerFactory::build(OptimizerModel& model)
{
    switch (model.type)
    {
    case dropout:
        return make_unique<Dropout>(model.value);
    default:
        throw InvalidArchitectureException("Optimizer type is not implemented.");
    }
}

void LayerOptimizerFactory::build(vector<unique_ptr<LayerOptimizer>>& optimizers, LayerModel& model)
{
    for (auto& optimizer : model.optimizers)
    {
        optimizers.push_back(build(optimizer));
    }
}