#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
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
        float learningRate;
        float momentum;

        StochasticGradientDescent() = default;
        StochasticGradientDescent(float learningRate, float momentum);
        StochasticGradientDescent(const StochasticGradientDescent& sgd) = default;
        ~StochasticGradientDescent() override = default;
        [[nodiscard]] std::shared_ptr<NeuralNetworkOptimizer> clone() const override;

        void updateWeights(SimpleNeuron& neuron, float error) const override;
        void updateWeights(RecurrentNeuron& neuron, float error) const override;

        [[nodiscard]] int isValid() override;

        bool operator==(const NeuralNetworkOptimizer& optimizer) const override;
        bool operator!=(const NeuralNetworkOptimizer& optimizer) const override;
    };

    template <class Archive>
    void StochasticGradientDescent::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<StochasticGradientDescent, NeuralNetworkOptimizer>();
        ar & boost::serialization::base_object<NeuralNetworkOptimizer>(*this);
        ar & this->learningRate;
        ar & this->momentum;
    }
}
