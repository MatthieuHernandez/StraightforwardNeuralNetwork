#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "SimpleLayer.hpp"
#include "neuron/RecurrentNeuron.hpp"

namespace snn::internal
{
class Recurrence final : public SimpleLayer<RecurrentNeuron>
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    public:
        Recurrence() = default;  // use restricted to Boost library only
        Recurrence(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Recurrence(const Recurrence&) = default;
        ~Recurrence() final = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> final;

        [[nodiscard]] auto summary() const -> std::string final;
};

template <class Archive>
void Recurrence::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Recurrence, SimpleLayer>();
    archive& boost::serialization::base_object<SimpleLayer>(*this);
}
}  // namespace snn::internal
