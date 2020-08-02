#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "SimpleLayer.hpp"
#include "../Optimizer.hpp"
#include "neuron/Neuron.hpp"

namespace snn::internal
{
    class FullyConnected final : public SimpleLayer<Neuron>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        FullyConnected() = default;  // use restricted to Boost library only
        FullyConnected(LayerModel& model, StochasticGradientDescent* optimizer);
        FullyConnected(const FullyConnected&) = default;
        ~FullyConnected() = default;
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;
    };

    template <class Archive>
    void FullyConnected::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<FullyConnected, SimpleLayer>();
        ar & boost::serialization::base_object<SimpleLayer>(*this);
    }
}
