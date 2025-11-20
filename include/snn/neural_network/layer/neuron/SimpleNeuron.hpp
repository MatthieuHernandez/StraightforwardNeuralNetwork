#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include "Neuron.hpp"

namespace snn::internal
{
class SimpleNeuron final : public Neuron
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    public:
        SimpleNeuron() = default;  // use restricted to Boost library only
        SimpleNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);

        [[nodiscard]] auto output(const std::vector<float>& inputs) -> float;
        void train();

        [[nodiscard]] auto isValid() const -> errorType;

        auto operator==(const SimpleNeuron& neuron) const -> bool;
        auto operator!=(const SimpleNeuron& neuron) const -> bool;
};

template <class Archive>
void SimpleNeuron::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<SimpleNeuron, Neuron>();
    archive& boost::serialization::base_object<Neuron>(*this);
}
}  // namespace snn::internal
