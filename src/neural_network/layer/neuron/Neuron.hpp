#pragma once
#include <cmath>
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "BaseNeuron.hpp"
#include "NeuronModel.hpp"
#include "../../Optimizer.hpp"
#include "activation_function/ActivationFunction.hpp"
#include "../../../tools/Tools.hpp"

namespace snn::internal
{
    template <class Derived>
    class Neuron : public BaseNeuron<Derived>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:

        int numberOfInputs;
        std::vector<float> weights;
        float bias;

        std::vector<float> previousDeltaWeights;
        std::vector<float> lastInputs;
        std::vector<float> errors;

        float sum = 0;

        activation activationFunction;
        std::shared_ptr<ActivationFunction> outputFunction;

        static float randomInitializeWeight(int numberOfInputs);

    public:
        Neuron() = default; // use restricted to Boost library only
        Neuron(NeuronModel model, StochasticGradientDescent* optimizer);
        Neuron(const Neuron& neuron) = default;
        ~Neuron() = default;

        StochasticGradientDescent* optimizer;

        [[nodiscard]] int isValid() const;

        [[nodiscard]] std::vector<float> getWeights() const;
        [[nodiscard]] int getNumberOfParameters() const;
        [[nodiscard]] int getNumberOfInputs() const;

        bool operator==(const Neuron& neuron) const;
        bool operator!=(const Neuron& neuron) const;
    };

    template <class Derived>
    template <class Archive>
    void Neuron<Derived>::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<Neuron, BaseNeuron<Derived>>();
        ar & boost::serialization::base_object<BaseNeuron<Derived>>(*this);
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


    template <class Derived>
    Neuron<Derived>::Neuron(NeuronModel model, StochasticGradientDescent* optimizer)
        : numberOfInputs(model.numberOfInputs),
          activationFunction(model.activationFunction),
          optimizer(optimizer)
    {
        this->previousDeltaWeights.resize(model.numberOfWeights, 0);
        this->lastInputs.resize(model.numberOfInputs, 0);
        this->errors.resize(model.numberOfInputs, 0);
        this->outputFunction = ActivationFunction::get(this->activationFunction);
        this->weights.resize(model.numberOfWeights);
        for (auto& w : this->weights)
        {
            w = randomInitializeWeight(model.numberOfWeights);
        }
        this->bias = 1.0f;
    }

    template <class Derived>
    float Neuron<Derived>::randomInitializeWeight(int numberOfWeights)
    {
        const float valueMax = 2.4f / sqrtf(static_cast<float>(numberOfWeights));
        return Tools::randomBetween(-valueMax, valueMax);
    }

    template <class Derived>
    int Neuron<Derived>::isValid() const
    {
        if (this->bias != 1.0f)
            return 301;

        if (this->weights.empty()
            || this->weights.size() > 1000000)
        {
            return 302;
        }
        for (auto& weight : this->weights)
            if (weight < -100000 || weight > 10000)
                return 303;

        return 0;
    }

    template <class Derived>
    std::vector<float> Neuron<Derived>::getWeights() const
    {
        return this->weights;
    }

    template <class Derived>
    int Neuron<Derived>::getNumberOfParameters() const
    {
        return static_cast<int>(this->weights.size());
    }

    template <class Derived>
    int Neuron<Derived>::getNumberOfInputs() const
    {
        return this->numberOfInputs;
    }

    template <class Derived>
    bool Neuron<Derived>::operator==(const Neuron& neuron) const
    {
        return this->numberOfInputs == neuron.numberOfInputs
            && this->weights == neuron.weights
            && this->bias == neuron.bias
            && this->previousDeltaWeights == neuron.previousDeltaWeights
            && this->lastInputs == neuron.lastInputs
            && this->errors == neuron.errors
            && this->sum == neuron.sum
            && this->activationFunction == neuron.activationFunction
            && this->outputFunction == neuron.outputFunction // not really good
            && *this->optimizer == *neuron.optimizer;
    }

    template <class Derived>
    bool Neuron<Derived>::operator!=(const Neuron& Neuron) const
    {
        return !(*this == Neuron);
    }
}
