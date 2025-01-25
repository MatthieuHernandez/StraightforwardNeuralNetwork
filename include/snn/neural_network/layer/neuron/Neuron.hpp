#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <queue>
#include <vector>

#include "../../../tools/Tools.hpp"
#include "../../optimizer/StochasticGradientDescent.hpp"
#include "BaseNeuron.hpp"
#include "CircularData.hpp"
#include "NeuronModel.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
class Neuron
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

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
        std::shared_ptr<NeuralNetworkOptimizer> optimizer = nullptr;

        static auto randomInitializeWeight(int numberOfInputs) -> float;

    public:
        Neuron() = default;  // use restricted to Boost library only
        Neuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Neuron(const Neuron& neuron) = default;
        ~Neuron() = default;

        std::shared_ptr<ActivationFunction> outputFunction;

        [[nodiscard]] auto isValid() const -> ErrorType;

        [[nodiscard]] auto getWeights() const -> std::vector<float>;
        void setWeights(std::vector<float> w);
        [[nodiscard]] auto getNumberOfParameters() const -> int;
        [[nodiscard]] auto getNumberOfInputs() const -> int;

        auto getOptimizer() const -> NeuralNetworkOptimizer*;
        void setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer);

        auto operator==(const Neuron& neuron) const -> bool;
        auto operator!=(const Neuron& neuron) const -> bool;
};

template <class Archive>
void Neuron::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    ar.template register_type<StochasticGradientDescent>();
    ar& this->optimizer;
    ar& this->numberOfInputs;
    ar& this->batchSize;
    ar& this->weights;
    ar& this->bias;
    ar& this->previousDeltaWeights;
    ar& this->lastInputs;
    ar& this->errors;
    ar& this->sum;
    ar& this->activationFunction;
    this->outputFunction = ActivationFunction::get(activationFunction);
}
}  // namespace snn::internal
