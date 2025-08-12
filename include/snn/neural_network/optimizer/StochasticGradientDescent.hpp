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
        void serialize(Archive& archive, uint32_t version);

    public:
        float learningRate{};
        float momentum{};

        StochasticGradientDescent() = default;
        StochasticGradientDescent(float learningRate, float momentum);
        StochasticGradientDescent(const StochasticGradientDescent& sgd) = default;
        ~StochasticGradientDescent() final = default;
        [[nodiscard]] auto clone() const -> std::shared_ptr<NeuralNetworkOptimizer> final;

        void updateWeights(SimpleNeuron& neuron) const final;
        void updateWeights(RecurrentNeuron& neuron) const final;

        [[nodiscard]] auto isValid() const -> errorType final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const NeuralNetworkOptimizer& optimizer) const -> bool final;
};

template <class Archive>
void StochasticGradientDescent::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<StochasticGradientDescent, NeuralNetworkOptimizer>();
    archive& boost::serialization::base_object<NeuralNetworkOptimizer>(*this);
    archive& this->learningRate;
    archive& this->momentum;
}
}  // namespace snn::internal
