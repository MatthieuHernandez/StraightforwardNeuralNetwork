#pragma once
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "NeuronModel.hpp"
#include "../../Optimizer.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class Neuron
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

    protected:

        int numberOfInputs;
        std::vector<float> weights;
        float bias;

        std::vector<float> previousDeltaWeights;
        std::vector<float> lastInputs;
        std::vector<float> errors;

        float sum = 0;

        activation activation;
        ActivationFunction* outputFunction;

        float randomInitializeWeight(int numberOfInputs) const;
        virtual void updateWeights(const std::vector<float>& inputs, float error);


    public:
        Neuron() = default; // use restricted to Boost library only
        Neuron(NeuronModel model, StochasticGradientDescent* optimizer);
        Neuron(const Neuron& neuron) = default;
        virtual ~Neuron() = default;

        StochasticGradientDescent* optimizer;

        [[nodiscard]] float output(const std::vector<float>& inputs);
        [[nodiscard]] virtual std::vector<float>& backOutput(float error);
        virtual void train(float error);

        [[nodiscard]] virtual int isValid() const;

        [[nodiscard]] std::vector<float> getWeights() const;
        [[nodiscard]] int getNumberOfParameters() const;

        void setWeights(const std::vector<float>& weights);

        [[nodiscard]] float getWeight(int w) const;
        void setWeight(int w, float weight);

        [[nodiscard]] float getBias() const;
        void setBias(float bias);

        [[nodiscard]] int getNumberOfInputs() const;

        virtual bool operator==(const Neuron& neuron) const;
        virtual bool operator!=(const Neuron& neuron) const;
    };

    template <class Archive>
    void Neuron::serialize(Archive& ar, const unsigned int version)
    {
        ar & this->numberOfInputs;
        ar & this->weights;
        ar & this->bias;
        ar & this->previousDeltaWeights;
        ar & this->lastInputs;
        ar & this->errors;
        ar & this->sum;
        ar & this->activation;
        this->outputFunction = ActivationFunction::get(activation);
        ar & this->optimizer;
    }
}
