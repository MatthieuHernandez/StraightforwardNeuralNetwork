#include <boost/serialization/export.hpp>
#include "Recurrence.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Recurrence)

Recurrence::Recurrence(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
     : SimpleLayer(model, optimizer)
{
}

unique_ptr<BaseLayer> Recurrence::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<Recurrence>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}