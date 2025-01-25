#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include "Neuron.hpp"

namespace snn::internal
{
class SimpleNeuron final : public Neuron
{
    private:
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

    public:
        SimpleNeuron() = default;  // use restricted to Boost library only
        SimpleNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        SimpleNeuron(const SimpleNeuron& neuron) = default;
        ~SimpleNeuron() = default;

        [[nodiscard]] auto output(const std::vector<float>& inputs) -> float;
        [[nodiscard]] auto backOutput(float error) -> std::vector<float>&;

        void train(float error);

        [[nodiscard]] auto isValid() const -> ErrorType;

        auto operator==(const SimpleNeuron& neuron) const -> bool;
        auto operator!=(const SimpleNeuron& neuron) const -> bool;
};

template <class Archive>
void SimpleNeuron::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<SimpleNeuron, Neuron>();
    ar& boost::serialization::base_object<Neuron>(*this);
}
}  // namespace snn::internal
