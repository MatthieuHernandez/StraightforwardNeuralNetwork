#include <boost/serialization/export.hpp>
#include "FullyConnected.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(FullyConnected)

FullyConnected::FullyConnected(LayerModel& model, StochasticGradientDescent* optimizer)
     : SimpleLayer(model, optimizer)
{
}

unique_ptr<BaseLayer> FullyConnected::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<FullyConnected>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}