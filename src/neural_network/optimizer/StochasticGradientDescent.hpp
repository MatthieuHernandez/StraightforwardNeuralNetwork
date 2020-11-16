#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include "NeuralNetworkOptimizer.hpp"


namespace snn::internal
{
    class StochasticGradientDescent final : public NeuralNetworkOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        float learningRate = 0.03f;
        float momentum = 0.0f;

        StochasticGradientDescent() = default;
        StochasticGradientDescent(const StochasticGradientDescent& sgd) = default;
        ~StochasticGradientDescent() = default;
        [[nodiscard]] std::shared_ptr<NeuralNetworkOptimizer> clone() const override;

        void updateWeight(const float& error, float& weight, float& previousDeltaWeight, const float& lastInput) const override;
        [[nodiscard]] int isValid() override;

        bool operator==(const NeuralNetworkOptimizer& sgd) const override;
        bool operator!=(const NeuralNetworkOptimizer& sgd) const override;
    };

    template <class Archive>
    void StochasticGradientDescent::serialize(Archive& ar, const unsigned int version)
    {
        ar & this->learningRate;
        ar & this->momentum;
    }
}
