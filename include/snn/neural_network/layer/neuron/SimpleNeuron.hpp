#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include "BaseNeuron.hpp"
#include "Neuron.hpp"

namespace snn::internal
{
class SimpleNeuron final : public Neuron
{
    private:
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    public:
        SimpleNeuron() = default;  // use restricted to Boost library only
        SimpleNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        SimpleNeuron(const SimpleNeuron& neuron) = default;
        ~SimpleNeuron() = default;

        [[nodiscard]] auto output(const std::vector<float>& inputs) -> float;
        [[nodiscard]] auto backOutput(float error) -> std::vector<float>&;
        void back(float error);
        void train();

        [[nodiscard]] auto isValid() const -> errorType;

        auto operator==(const SimpleNeuron& neuron) const -> bool;
};
static_assert(BaseNeuron<SimpleNeuron>);

template <class Archive>
void SimpleNeuron::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<SimpleNeuron, Neuron>();
    archive& boost::serialization::base_object<Neuron>(*this);
}
}  // namespace snn::internal
