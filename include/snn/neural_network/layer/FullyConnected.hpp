#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "SimpleLayer.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
class FullyConnected final : public SimpleLayer<SimpleNeuron>
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

    public:
        FullyConnected() = default;  // use restricted to Boost library only
        FullyConnected(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        FullyConnected(const FullyConnected&) = default;
        ~FullyConnected() = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> override;

        [[nodiscard]] auto summary() const -> std::string override;
};

template <class Archive>
void FullyConnected::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<FullyConnected, SimpleLayer>();
    ar& boost::serialization::base_object<SimpleLayer>(*this);
}
}  // namespace snn::internal
