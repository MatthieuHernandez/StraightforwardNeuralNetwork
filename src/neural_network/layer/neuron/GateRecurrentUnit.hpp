#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "BaseNeuron.hpp"
#include "RecurrentNeuron.hpp"

namespace snn::internal
{
    class GateRecurrentUnit final : public BaseNeuron
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

       friend class RecurrentNeuron;

        int numberOfInputs;

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
        GateRecurrentUnit() = default; // use restricted to Boost library only
        GateRecurrentUnit(NeuronModel model, StochasticGradientDescent* optimizer);
        GateRecurrentUnit(const GateRecurrentUnit& recurrentNeuron) = default;
        ~GateRecurrentUnit() = default;

        StochasticGradientDescent* optimizer;

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
    void GateRecurrentUnit::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<GateRecurrentUnit, BaseNeuron>();
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
