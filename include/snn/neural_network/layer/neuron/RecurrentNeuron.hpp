#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include "Neuron.hpp"

namespace snn::internal
{
class RecurrentNeuron final : public Neuron
{
    private:
        friend class GatedRecurrentUnit;
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

        float lastOutput = 0;
        float previousOutput = 0;
        float recurrentError = 0;
        float previousSum = 0;

        void reset();

    public:
        RecurrentNeuron() = default;  // use restricted to Boost library only
        RecurrentNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        RecurrentNeuron(const RecurrentNeuron& recurrentNeuron) = default;
        ~RecurrentNeuron() = default;

        [[nodiscard]] auto output(const std::vector<float>& inputs, bool reset) -> float;
        [[nodiscard]] auto backOutput(float error) -> std::vector<float>&;
        void train(float error);

        [[nodiscard]] auto isValid() const -> ErrorType;

        auto operator==(const RecurrentNeuron& neuron) const -> bool;
        auto operator!=(const RecurrentNeuron& neuron) const -> bool;
};

template <class Archive>
void RecurrentNeuron::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<RecurrentNeuron, Neuron>();
    ar& boost::serialization::base_object<Neuron>(*this);
    ar& this->lastOutput;
    ar& this->previousOutput;
    ar& this->recurrentError;
    ar& this->previousSum;
}
}  // namespace snn::internal
