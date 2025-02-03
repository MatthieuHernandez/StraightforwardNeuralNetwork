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
        void serialize(Archive& archive, uint32_t version);

    public:
        GruLayer() = default;  // use restricted to Boost library only
        GruLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        GruLayer(const GruLayer&) = default;
        ~GruLayer() final = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> final;

        [[nodiscard]] auto summary() const -> std::string final;
};

template <class Archive>
void GruLayer::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<GruLayer, SimpleLayer>();
    archive& boost::serialization::base_object<SimpleLayer>(*this);
}
}  // namespace snn::internal