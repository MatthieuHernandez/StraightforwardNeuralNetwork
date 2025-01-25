#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "NeuralNetworkOptimizer.hpp"

namespace snn::internal
{
class StochasticGradientDescent final : public NeuralNetworkOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

    public:
        float learningRate{};
        float momentum{};

        StochasticGradientDescent() = default;
        StochasticGradientDescent(float learningRate, float momentum);
        StochasticGradientDescent(const StochasticGradientDescent& sgd) = default;
        ~StochasticGradientDescent() = default;
        [[nodiscard]] auto clone() const -> std::shared_ptr<NeuralNetworkOptimizer> override;

        void updateWeights(SimpleNeuron& neuron, float error) const override;
        void updateWeights(RecurrentNeuron& neuron, float error) const override;

        [[nodiscard]] auto isValid() const -> ErrorType override;

        [[nodiscard]] auto summary() const -> std::string override;

        auto operator==(const NeuralNetworkOptimizer& optimizer) const -> bool override;
        auto operator!=(const NeuralNetworkOptimizer& optimizer) const -> bool override;
};

template <class Archive>
void StochasticGradientDescent::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<StochasticGradientDescent, NeuralNetworkOptimizer>();
    ar& boost::serialization::base_object<NeuralNetworkOptimizer>(*this);
    ar& this->learningRate;
    ar& this->momentum;
}
}  // namespace snn::internal
