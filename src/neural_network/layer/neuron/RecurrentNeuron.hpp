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

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        float lastOutput = 0;
        float previousOutput = 0;
        float recurrentError = 0;
        float previousSum = 0;

        void reset();
        void updateWeights(const float error) override;

    public:
        RecurrentNeuron() = default; // use restricted to Boost library only
        RecurrentNeuron(NeuronModel model, StochasticGradientDescent* optimizer);
        RecurrentNeuron(const RecurrentNeuron& recurrentNeuron) = default;
        ~RecurrentNeuron() = default;

        [[nodiscard]] virtual float output(const std::vector<float>& inputs, bool reset) override;
        [[nodiscard]] std::vector<float>& backOutput(float error) override;
        void train(float error) override;

        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseNeuron& neuron) const override;
        bool operator!=(const BaseNeuron& neuron) const override;
    };

    template <class Archive>
    void RecurrentNeuron::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<RecurrentNeuron, Neuron>();
        ar & boost::serialization::base_object<Neuron>(*this);
        ar & this->lastOutput;
        ar & this->previousOutput;
        ar & this->recurrentError;
        ar & this->previousSum;
    }
}
