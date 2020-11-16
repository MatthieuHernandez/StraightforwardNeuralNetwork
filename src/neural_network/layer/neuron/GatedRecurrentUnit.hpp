#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "BaseNeuron.hpp"
#include "RecurrentNeuron.hpp"

namespace snn::internal
{
    class GatedRecurrentUnit final : public BaseNeuron<GatedRecurrentUnit>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

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
        GatedRecurrentUnit() = default; // use restricted to Boost library only
        GatedRecurrentUnit(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        GatedRecurrentUnit(const GatedRecurrentUnit& recurrentNeuron) = default;
        ~GatedRecurrentUnit() = default;

        [[nodiscard]] float output(const std::vector<float>& inputs, bool reset);
        [[nodiscard]] std::vector<float>& backOutput(float error);
        void train(float error);

        [[nodiscard]] std::vector<float> getWeights() const;
        [[nodiscard]] int getNumberOfParameters() const;
        [[nodiscard]] int getNumberOfInputs() const;

        [[nodiscard]] int isValid() const;

        bool operator==(const GatedRecurrentUnit& neuron) const;
        bool operator!=(const GatedRecurrentUnit& neuron) const;
    };

    template <class Archive>
    void GatedRecurrentUnit::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<GatedRecurrentUnit, BaseNeuron<GatedRecurrentUnit>>();
        ar & boost::serialization::base_object<BaseNeuron<GatedRecurrentUnit>>(*this);
        ar & this->numberOfInputs;
        ar & this->previousOutput;
        ar & this->recurrentError;
        ar & this->updateGateOutput;
        ar & this->outputGateOutput;
        ar & this->resetGate;
        ar & this->updateGate;
        ar & this->outputGate;
    }
}
