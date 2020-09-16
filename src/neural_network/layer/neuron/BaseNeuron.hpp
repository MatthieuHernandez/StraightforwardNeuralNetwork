#pragma once
#include <vector>
#include <boost/serialization/access.hpp>
#include "../../tools/ExtendedExpection.hpp"

namespace snn::internal
{
    class BaseNeuron
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:

        virtual void updateWeights(const float error) = 0;

    public:
        [[nodiscard]] virtual float output(const std::vector<float>& inputs) { throw NotImplementedException(); }

        [[nodiscard]] virtual float output(const std::vector<float>& inputs, bool reset) { throw NotImplementedException(); }

        [[nodiscard]] virtual std::vector<float>& backOutput(float error) = 0;
        virtual void train(float error) = 0;

        [[nodiscard]] virtual int isValid() const = 0;

        [[nodiscard]] virtual std::vector<float> getWeights() const = 0;
        [[nodiscard]] virtual int getNumberOfParameters() const = 0;
        [[nodiscard]] virtual int getNumberOfInputs() const = 0;

        virtual bool operator==(const BaseNeuron& neuron) const
        {
            return typeid(*this).hash_code() == typeid(neuron).hash_code();
        }

        virtual bool operator!=(const BaseNeuron& neuron) const
        {
            return !(*this == neuron);
        }
    };

    template <class Archive>
    void BaseNeuron::serialize(Archive& ar, unsigned version)
    {
    }
}
