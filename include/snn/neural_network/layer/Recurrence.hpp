#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/StochasticGradientDescent.hpp"
#include "SimpleLayer.hpp"
#include "neuron/RecurrentNeuron.hpp"

namespace snn::internal
{
class Recurrence final : public SimpleLayer<RecurrentNeuron>
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

    public:
        Recurrence() = default;  // use restricted to Boost library only
        Recurrence(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Recurrence(const Recurrence&) = default;
        ~Recurrence() = default;
        auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer> override;

        [[nodiscard]] auto summary() const -> std::string override;
};

template <class Archive>
void Recurrence::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Recurrence, SimpleLayer>();
    ar& boost::serialization::base_object<SimpleLayer>(*this);
}
}  // namespace snn::internal
