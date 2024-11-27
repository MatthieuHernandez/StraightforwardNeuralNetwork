#pragma once
#include <memory>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/access.hpp>

namespace snn::internal
{
    class SimpleNeuron;
    class RecurrentNeuron;

    class NeuralNetworkOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize([[maybe_unused]] Archive& ar, [[maybe_unused]] const unsigned version) {}

    public:
        NeuralNetworkOptimizer() = default;
        virtual ~NeuralNetworkOptimizer() = default;
        [[nodiscard]] virtual std::shared_ptr<NeuralNetworkOptimizer> clone() const = 0;

        virtual void updateWeights(SimpleNeuron& neuron, float error) const = 0;
        virtual void updateWeights(RecurrentNeuron& neuron, float error) const = 0;

        [[nodiscard]] virtual int isValid() = 0;

        [[nodiscard]] virtual std::string summary() const = 0;

        virtual bool operator==(const NeuralNetworkOptimizer& optimizer) const;
        virtual bool operator!=(const NeuralNetworkOptimizer& optimizer) const;
    };
}
