#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Neuron.hpp"

namespace snn::internal
{
    class SimpleNeuron final : public Neuron
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

    public:
        SimpleNeuron() = default; // use restricted to Boost library only
        SimpleNeuron(NeuronModel model, StochasticGradientDescent* optimizer);
        SimpleNeuron(const SimpleNeuron& neuron) = default;
        ~SimpleNeuron() = default;

        [[nodiscard]] int isValid() const override;

        bool operator==(const Neuron& neuron) const override;
        bool operator!=(const Neuron& neuron) const override;
    };

    template <class Archive>
    void SimpleNeuron::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<SimpleNeuron, Neuron>();
        ar & boost::serialization::base_object<Neuron>(*this);
    }
}
