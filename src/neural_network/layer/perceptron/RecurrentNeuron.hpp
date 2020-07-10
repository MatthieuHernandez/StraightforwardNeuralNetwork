#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Neuron.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class RecurrentNeuron final : public Neuron
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

    public:
        RecurrentNeuron() = default; // use restricted to Boost library only
        RecurrentNeuron(int numberOfInputs, activationFunction activation, StochasticGradientDescent* optimizer);
        RecurrentNeuron(const RecurrentNeuron& recurrentNeuron) = default;
        ~RecurrentNeuron() = default;

        [[nodiscard]] int isValid() const override;

        bool operator==(const Neuron& neuron) const override;
        bool operator!=(const Neuron& neuron) const override;
    };

    template <class Archive>
    void RecurrentNeuron::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<RecurrentNeuron, Neuron>();
        ar & boost::serialization::base_object<Neuron>(*this);
    }
}
