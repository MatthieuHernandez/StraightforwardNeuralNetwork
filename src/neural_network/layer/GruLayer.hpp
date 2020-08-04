#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "SimpleLayer.hpp"
#include "../Optimizer.hpp"
#include "neuron/GateRecurrentUnit.hpp"

namespace snn::internal
{
    class GruLayer final : public SimpleLayer<GateRecurrentUnit>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        GruLayer() = default;  // use restricted to Boost library only
        GruLayer(LayerModel& model, StochasticGradientDescent* optimizer);
        GruLayer(const GruLayer&) = default;
        ~GruLayer() = default;
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;
    };

    template <class Archive>
    void GruLayer::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<GruLayer, SimpleLayer>();
        ar & boost::serialization::base_object<SimpleLayer>(*this);
    }
}