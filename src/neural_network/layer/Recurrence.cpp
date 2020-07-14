#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "Recurrence.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

BOOST_CLASS_EXPORT(Recurrence)

Recurrence::Recurrence(LayerModel& model, StochasticGradientDescent* optimizer)
     : SimpleLayer(model, optimizer)
{
}

unique_ptr<BaseLayer> Recurrence::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<Recurrence>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}