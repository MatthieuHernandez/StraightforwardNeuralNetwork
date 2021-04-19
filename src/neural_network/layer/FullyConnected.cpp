#include <boost/serialization/export.hpp>
#include "FullyConnected.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(FullyConnected)

FullyConnected::FullyConnected(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : SimpleLayer(model, optimizer)
{
}

unique_ptr<BaseLayer> FullyConnected::clone(shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<FullyConnected>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}
