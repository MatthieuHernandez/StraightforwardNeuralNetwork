#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include "../../optimizer/StochasticGradientDescent.hpp"
#include "Circular.hpp"
#include "NeuronModel.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
class Neuron
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    protected:
        int numberOfInputs{};
        int batchSize{};
        std::vector<float> weights;
        float bias{};

        Circular<std::vector<float>> previousDeltaWeights;
        Circular<std::vector<float>> lastInputs;
        Circular<float> lastErrors;
        std::vector<float> errors;

        float sum = 0;

        activation activationFunction{};
        std::shared_ptr<NeuralNetworkOptimizer> optimizer = nullptr;

        static auto randomInitializeWeight(int numberOfWeights) -> float;

    public:
        Neuron() = default;  // use restricted to Boost library only
        Neuron(Neuron&&) = delete;
        auto operator=(const Neuron&) -> Neuron& = default;
        auto operator=(Neuron&&) -> Neuron& = delete;
        Neuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Neuron(const Neuron& neuron) = default;
        ~Neuron() = default;

        std::shared_ptr<ActivationFunction> outputFunction;

        [[nodiscard]] auto isValid() const -> errorType;

        [[nodiscard]] auto getWeights() const -> std::vector<float>;
        void setWeights(std::vector<float> w);
        [[nodiscard]] auto getNumberOfParameters() const -> int;
        [[nodiscard]] auto getNumberOfInputs() const -> int;

        [[nodiscard]] auto getOptimizer() const -> NeuralNetworkOptimizer*;
        void setOptimizer(std::shared_ptr<NeuralNetworkOptimizer> newOptimizer);

        auto operator==(const Neuron& neuron) const -> bool;
        auto operator!=(const Neuron& neuron) const -> bool;
};

template <class Archive>
void Neuron::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    archive.template register_type<StochasticGradientDescent>();
    archive& this->optimizer;
    archive& this->numberOfInputs;
    archive& this->batchSize;
    archive& this->weights;
    archive& this->bias;
    archive& this->previousDeltaWeights;
    archive& this->lastInputs;
    archive& this->lastErrors;
    archive& this->errors;
    archive& this->sum;
    archive& this->activationFunction;
    this->outputFunction = ActivationFunction::get(activationFunction);
}
}  // namespace snn::internal
