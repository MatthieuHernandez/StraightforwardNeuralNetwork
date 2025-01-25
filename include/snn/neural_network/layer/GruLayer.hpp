#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "SimpleLayer.hpp"
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
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> override;

        [[nodiscard]] auto summary() const -> std::string override;
};

template <class Archive>
void GruLayer::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<GruLayer, SimpleLayer>();
    ar& boost::serialization::base_object<SimpleLayer>(*this);
}
}  // namespace snn::internal