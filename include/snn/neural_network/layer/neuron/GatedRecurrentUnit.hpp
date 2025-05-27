#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include "BaseNeuron.hpp"
#include "RecurrentNeuron.hpp"

namespace snn::internal
{
class GatedRecurrentUnit final
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        friend class RecurrentNeuron;

        std::vector<float> errors;

        int numberOfInputs{};

        float previousOutput = 0;
        float recurrentError = 0;
        float updateGateOutput = 0;
        float outputGateOutput = 0;

        RecurrentNeuron resetGate;
        RecurrentNeuron updateGate;
        RecurrentNeuron outputGate;

        void reset();

    public:
        GatedRecurrentUnit() = default;  // use restricted to Boost library only
        GatedRecurrentUnit(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        GatedRecurrentUnit(const GatedRecurrentUnit& recurrentNeuron) = default;
        ~GatedRecurrentUnit() = default;

        [[nodiscard]] auto output(const std::vector<float>& inputs, bool reset) -> float;
        [[nodiscard]] auto backOutput(float error) -> std::vector<float>&;
        void back(float error);
        void train();

        [[nodiscard]] auto getWeights() const -> std::vector<float>;
        [[nodiscard]] auto getNumberOfParameters() const -> int;
        [[nodiscard]] auto getNumberOfInputs() const -> int;

        [[nodiscard]] auto isValid() const -> errorType;

        [[nodiscard]] auto getOptimizer() const -> NeuralNetworkOptimizer*;
        void setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer);

        auto operator==(const GatedRecurrentUnit& neuron) const -> bool;
        auto operator!=(const GatedRecurrentUnit& neuron) const -> bool;
};

template <class Archive>
void GatedRecurrentUnit::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    archive& this->errors;
    archive& this->numberOfInputs;
    archive& this->previousOutput;
    archive& this->recurrentError;
    archive& this->updateGateOutput;
    archive& this->outputGateOutput;
    archive& this->resetGate;
    archive& this->updateGate;
    archive& this->outputGate;
}
}  // namespace snn::internal
