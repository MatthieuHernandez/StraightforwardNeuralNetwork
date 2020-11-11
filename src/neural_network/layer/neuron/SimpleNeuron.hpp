#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Neuron.hpp"

namespace snn::internal
{
    class SimpleNeuron final : public Neuron<SimpleNeuron>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        void updateWeights(const float error);

    public:
        SimpleNeuron() = default; // use restricted to Boost library only
        SimpleNeuron(NeuronModel model, StochasticGradientDescent* optimizer);
        SimpleNeuron(const SimpleNeuron& neuron) = default;
        ~SimpleNeuron() = default;

        [[nodiscard]] float output(const std::vector<float>& inputs);
        [[nodiscard]] std::vector<float>& backOutput(float error);
        void train(float error);

        [[nodiscard]] int isValid() const;

        bool operator==(const SimpleNeuron& neuron) const;
        bool operator!=(const SimpleNeuron& neuron) const;
    };

    template <class Archive>
    void SimpleNeuron::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<SimpleNeuron, Neuron>();
        ar & boost::serialization::base_object<Neuron>(*this);
    }
}
