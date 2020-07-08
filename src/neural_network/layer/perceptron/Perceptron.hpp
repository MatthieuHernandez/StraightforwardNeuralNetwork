#pragma once
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "Neuron.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class Perceptron : public Neuron
    {
    private:
        std::vector<float> weights;
        float bias;

        std::vector<float> previousDeltaWeights;
        std::vector<float> lastInputs;
        std::vector<float> errors;

        float lastOutput = 0;

        activationFunction activation;
        ActivationFunction* outputFunction;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

    public:
        Perceptron() = default; // use restricted to Boost library only
        Perceptron(int numberOfInputs, activationFunction activation, StochasticGradientDescent* optimizer);
        Perceptron(const Perceptron& perceptron) = default;
        ~Perceptron() = default;

        [[nodiscard]] int isValid() const;

        bool operator==(const Perceptron& perceptron) const;
        bool operator!=(const Perceptron& perceptron) const;
    };

    template <class Archive>
    void Perceptron::serialize(Archive& ar, const unsigned int version)
    {
        ar & this->weights;
        ar & this->bias;
        ar & this->previousDeltaWeights;
        ar & this->lastInputs;
        ar & this->errors;
        ar & this->lastOutput;
        ar & this->activation;
        this->outputFunction = ActivationFunction::get(activation);
        ar & this->optimizer;
    }
}
