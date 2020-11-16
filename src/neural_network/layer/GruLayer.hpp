#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "SimpleLayer.hpp"
#include "../optimizer/StochasticGradientDescent.hpp"
#include "neuron/GatedRecurrentUnit.hpp"

namespace snn::internal
{
    class GruLayer final : public SimpleLayer<GatedRecurrentUnit>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        GruLayer() = default;  // use restricted to Boost library only
        GruLayer(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer);
        GruLayer(const GruLayer&) = default;
        ~GruLayer() = default;
        std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;
    };

    template <class Archive>
    void GruLayer::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<GruLayer, SimpleLayer>();
        ar & boost::serialization::base_object<SimpleLayer>(*this);
    }
}