#pragma once
#include <vector>
#include <queue>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include "BaseNeuron.hpp"
#include "NeuronModel.hpp"
#include "CircularData.hpp"
#include "../../optimizer/StochasticGradientDescent.hpp"
#include "activation_function/ActivationFunction.hpp"
#include "../../../tools/Tools.hpp"

namespace snn::internal
{
    class Neuron
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        int numberOfInputs;
        int batchSize;
        std::vector<float> weights;
        float bias;

        CircularData previousDeltaWeights;
        CircularData lastInputs;
        std::vector<float> errors;

        float sum = 0;

        activation activationFunction;
        std::shared_ptr<ActivationFunction> outputFunction;
        std::shared_ptr<NeuralNetworkOptimizer> optimizer = nullptr;

        static float randomInitializeWeight(int numberOfInputs);

    public:
        Neuron() = default; // use restricted to Boost library only
        Neuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Neuron(const Neuron& neuron) = default;
        ~Neuron() = default;

        [[nodiscard]] int isValid() const;

        [[nodiscard]] std::vector<float> getWeights() const;
        void setWeights(std::vector<float> w);
        [[nodiscard]] int getNumberOfParameters() const;
        [[nodiscard]] int getNumberOfInputs() const;

        NeuralNetworkOptimizer* getOptimizer() const;
        void setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer);

        bool operator==(const Neuron& neuron) const;
        bool operator!=(const Neuron& neuron) const;
    };

    template <class Archive>
    void Neuron::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        ar.template register_type<StochasticGradientDescent>();
        ar & this->optimizer;
        ar & this->numberOfInputs;
        ar & this->batchSize;
        ar & this->weights;
        ar & this->bias;
        ar & this->previousDeltaWeights;
        ar & this->lastInputs;
        ar & this->errors;
        ar & this->sum;
        ar & this->activationFunction;
        this->outputFunction = ActivationFunction::get(activationFunction);
    }
}
