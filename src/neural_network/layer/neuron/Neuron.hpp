#pragma once
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "BaseNeuron.hpp"
#include "NeuronModel.hpp"
#include "../../Optimizer.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class Neuron : public BaseNeuron
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

        activation activationFunction;
        ActivationFunction* outputFunction;

        float randomInitializeWeight(int numberOfInputs) const;

    public:
        Neuron() = default; // use restricted to Boost library only
        Neuron(NeuronModel model, StochasticGradientDescent* optimizer);
        Neuron(const Neuron& neuron) = default;
        virtual ~Neuron() = default;

        StochasticGradientDescent* optimizer;

        [[nodiscard]] int isValid() const override;

        [[nodiscard]] std::vector<float> getWeights() const override;
        [[nodiscard]] int getNumberOfParameters() const override;
        [[nodiscard]] int getNumberOfInputs() const override;

        virtual bool operator==(const BaseNeuron& neuron) const override;
        virtual bool operator!=(const BaseNeuron& neuron) const override;
    };

    template <class Archive>
    void Neuron::serialize(Archive& ar, const unsigned int)
    {
         boost::serialization::void_cast_register<Neuron, BaseNeuron>();
        ar & boost::serialization::base_object<BaseNeuron>(*this);
        ar & this->numberOfInputs;
        ar & this->weights;
        ar & this->bias;
        ar & this->previousDeltaWeights;
        ar & this->lastInputs;
        ar & this->errors;
        ar & this->sum;
        ar & this->activationFunction;
        this->outputFunction = ActivationFunction::get(activationFunction);
        ar & this->optimizer;
    }
}
