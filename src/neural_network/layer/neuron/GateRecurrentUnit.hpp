#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Neuron.hpp"

namespace snn::internal
{
    class GateRecurrentUnit final : public Neuron
    {
            private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

        float lastOutput = 0;
        float previousOutput = 0;
        float recurrentError = 0;
        float previousSum = 0;

        void reset();
        void updateWeights(const std::vector<float>& inputs, float error) override;

        void forgetGateOutput(const std::vector<float>& inputs);
        void updateGateOutput(const std::vector<float>& inputs);

        std::vector<float>& forgetGateBackOutput(const std::vector<float>& inputs);
        std::vector<float>& updateGateBackOutput(const std::vector<float>& inputs);

    public:
        GateRecurrentUnit() = default; // use restricted to Boost library only
        GateRecurrentUnit(NeuronModel model, StochasticGradientDescent* optimizer);
        GateRecurrentUnit(const GateRecurrentUnit& recurrentNeuron) = default;
        ~GateRecurrentUnit() = default;

        [[nodiscard]] float output(const std::vector<float>& inputs, bool reset);

        [[nodiscard]] int isValid() const override;

        bool operator==(const Neuron& neuron) const override;
        bool operator!=(const Neuron& neuron) const override;
        [[nodiscard]] std::vector<float>& backOutput(float error) override;
        void train(float error) override;
    };

    template <class Archive>
    void GateRecurrentUnit::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<GateRecurrentUnit, Neuron>();
        ar & boost::serialization::base_object<Neuron>(*this);
    }
}