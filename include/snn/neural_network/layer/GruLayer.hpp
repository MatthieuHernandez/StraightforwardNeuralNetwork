#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "SimpleLayer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
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
        GruLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        GruLayer(const GruLayer&) = default;
        ~GruLayer() = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;
    };

    template <class Archive>
    void GruLayer::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<GruLayer, SimpleLayer>();
        ar & boost::serialization::base_object<SimpleLayer>(*this);
    }
}