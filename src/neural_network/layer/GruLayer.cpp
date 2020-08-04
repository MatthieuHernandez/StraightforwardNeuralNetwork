#include <boost/serialization/export.hpp>
#include "GruLayer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(GruLayer)

GruLayer::GruLayer(LayerModel& model, StochasticGradientDescent* optimizer)
     : SimpleLayer(model, optimizer)
{
}

unique_ptr<BaseLayer> GruLayer::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<GruLayer>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}