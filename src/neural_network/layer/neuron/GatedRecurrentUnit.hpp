#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "BaseNeuron.hpp"
#include "RecurrentNeuron.hpp"

namespace snn::internal
{
    class GatedRecurrentUnit final : public BaseNeuron
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
        void updateWeights(const float error) override;

    public:
        GatedRecurrentUnit() = default; // use restricted to Boost library only
        GatedRecurrentUnit(NeuronModel model, StochasticGradientDescent* optimizer);
        GatedRecurrentUnit(const GatedRecurrentUnit& recurrentNeuron) = default;
        ~GatedRecurrentUnit() = default;

        StochasticGradientDescent* optimizer{};

        [[nodiscard]] float output(const std::vector<float>& inputs, bool reset) override;
        [[nodiscard]] std::vector<float>& backOutput(float error) override;
        void train(float error) override;

        [[nodiscard]] std::vector<float> getWeights() const override;
        [[nodiscard]] int getNumberOfParameters() const override;
        [[nodiscard]] int getNumberOfInputs() const override;

        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseNeuron& neuron) const override;
        bool operator!=(const BaseNeuron& neuron) const override;
    };

    template <class Archive>
    void GatedRecurrentUnit::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<GatedRecurrentUnit, BaseNeuron>();
        ar & boost::serialization::base_object<BaseNeuron>(*this);
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
