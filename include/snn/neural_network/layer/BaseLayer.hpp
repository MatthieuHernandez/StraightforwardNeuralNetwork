#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <memory>

#include "../../tools/Error.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "neuron/BaseNeuron.hpp"

namespace snn::internal
{
class BaseLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize([[maybe_unused]] Archive& ar, [[maybe_unused]] const uint32_t version)
        {
        }

    public:
        virtual ~BaseLayer() = default;
        [[nodiscard]] virtual auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> = 0;

        [[nodiscard]] virtual auto getNeuron(int index) -> void* = 0;
        [[nodiscard]] virtual auto getAverageOfAbsNeuronWeights() const -> float = 0;
        [[nodiscard]] virtual auto getAverageOfSquareNeuronWeights() const -> float = 0;
        [[nodiscard]] virtual auto getNumberOfInputs() const -> int = 0;
        [[nodiscard]] virtual auto getNumberOfNeurons() const -> int = 0;
        [[nodiscard]] virtual auto getNumberOfParameters() const -> int = 0;
        [[nodiscard]] virtual auto getShapeOfInput() const -> std::vector<int> = 0;
        [[nodiscard]] virtual auto getShapeOfOutput() const -> std::vector<int> = 0;

        [[nodiscard]] virtual auto output(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> = 0;
        [[nodiscard]] virtual auto outputForTraining(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> = 0;
        [[nodiscard]] virtual auto backOutput(std::vector<float>& inputErrors) -> std::vector<float> = 0;
        virtual void train(std::vector<float>& inputErrors) = 0;

        [[nodiscard]] virtual auto isValid() const -> errorType = 0;

        [[nodiscard]] virtual auto summary() const -> std::string = 0;

        virtual auto operator==(const BaseLayer& layer) const -> bool = 0;
        virtual auto operator!=(const BaseLayer& layer) const -> bool = 0;
};
}  // namespace snn::internal
