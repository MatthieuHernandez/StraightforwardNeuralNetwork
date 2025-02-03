#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <memory>

#include "../../tools/Error.hpp"

namespace snn::internal
{
class SimpleNeuron;
class RecurrentNeuron;

class NeuralNetworkOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize([[maybe_unused]] Archive& ar, [[maybe_unused]] const uint32_t version)
        {
        }

    public:
        NeuralNetworkOptimizer() = default;
        virtual ~NeuralNetworkOptimizer() = default;
        [[nodiscard]] virtual auto clone() const -> std::shared_ptr<NeuralNetworkOptimizer> = 0;

        virtual void updateWeights(SimpleNeuron& neuron, float error) const = 0;
        virtual void updateWeights(RecurrentNeuron& neuron, float error) const = 0;

        [[nodiscard]] virtual auto isValid() const -> errorType = 0;

        [[nodiscard]] virtual auto summary() const -> std::string = 0;

        virtual auto operator==(const NeuralNetworkOptimizer& optimizer) const -> bool;
        virtual auto operator!=(const NeuralNetworkOptimizer& optimizer) const -> bool;
};
}  // namespace snn::internal
