#include "LayerOptimizerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "Dropout.hpp";

using namespace std;
using namespace snn;
using namespace internal;

std::unique_ptr<LayerOptimizer> LayerOptimizerFactory::build(OptimizerModel& model, LayerModel& layerModel)
{
    switch (model.type)
    {
    case dropout:
        return make_unique<Dropout>(model.value);
    default:
        throw InvalidArchitectureException("Optimizer type is not implemented.");
    }
}

void LayerOptimizerFactory::build(std::vector<std::unique_ptr<LayerOptimizer>>& optimizers,
                                        std::vector<OptimizerModel>& models, LayerModel& model)
{
    for (int o = 0; o < model.optimizers.size(); ++o)
    {
        optimizers.push_back(build(model.optimizers[o], model));
    }
}
