#pragma once
#include <memory>
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "../../Optimizer.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class Perceptron
    {
    private:
        std::vector<float> weights;
        float bias;

        std::vector<float> previousDeltaWeights;
        std::vector<float> lastInputs;
        std::vector<float> errors;

        float lastOutput = 0;

        //StochasticGradientDescent* optimizer;

        activationFunction activation;
        ActivationFunction* outputFunction;

        float randomInitializeWeight(int numberOfInputs) const;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

    public:
        Perceptron() = default; // use restricted to Boost library only
        Perceptron(int numberOfInputs, activationFunction activation, StochasticGradientDescent* optimizer);
        Perceptron(const Perceptron& perceptron) = default;
        ~Perceptron() = default;

        StochasticGradientDescent* optimizer;

        std::vector<float>& backOutput(float error);
        float output(const std::vector<float>& inputs);
        void train(const std::vector<float>& inputs, float error);

        [[nodiscard]] int isValid() const;

        [[nodiscard]] std::vector<float> getWeights() const;
        [[nodiscard]] int getNumberOfParameters() const;

        void setWeights(const std::vector<float>& weights);

        [[nodiscard]] float getWeight(int w) const;
        void setWeight(int w, float weight);

        [[nodiscard]] float getBias() const;
        void setBias(float bias);

        [[nodiscard]] int getNumberOfInputs() const;

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
