#pragma once
#include <boost/serialization/access.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../tools/Error.hpp"
#include "../optimizer/Dropout.hpp"
#include "../optimizer/ErrorMultiplier.hpp"
#include "../optimizer/L1Regularization.hpp"
#include "../optimizer/L2Regularization.hpp"
#include "../optimizer/LayerOptimizer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "../optimizer/Softmax.hpp"
#include "BaseLayer.hpp"
#include "LayerModel.hpp"
#include "neuron/BaseNeuron.hpp"
#include "neuron/GatedRecurrentUnit.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
template <BaseNeuron N>
class Layer : public BaseLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    protected:
        int numberOfInputs{};

        [[nodiscard]] virtual auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> = 0;
        [[nodiscard]] virtual auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> = 0;
        virtual void computeTrain(std::vector<float>& inputErrors) = 0;

    public:
        Layer() = default;  // use restricted to Boost library only
        Layer(Layer&&) = delete;
        auto operator=(const Layer&) -> Layer& = delete;
        auto operator=(Layer&&) -> Layer& = delete;
        Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Layer(const Layer& layer);
        ~Layer() override = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> override = 0;

        std::vector<N> neurons{};
        std::vector<std::unique_ptr<LayerOptimizer>> optimizers;

        auto output(const std::vector<float>& inputs, bool temporalReset) -> std::vector<float> final;
        auto outputForTraining(const std::vector<float>& inputs, bool temporalReset) -> std::vector<float> final;
        auto backOutput(std::vector<float>& inputErrors) -> std::vector<float> final;

        [[nodiscard]] auto getNeuron(int index) -> void* final;
        [[nodiscard]] auto getAverageOfAbsNeuronWeights() const -> float final;
        [[nodiscard]] auto getAverageOfSquareNeuronWeights() const -> float final;
        [[nodiscard]] auto getNumberOfInputs() const -> int final;
        [[nodiscard]] auto getNumberOfNeurons() const -> int final;
        [[nodiscard]] auto getNumberOfParameters() const -> int final;
        [[nodiscard]] auto getShapeOfInput() const -> std::vector<int> override = 0;
        [[nodiscard]] auto getShapeOfOutput() const -> std::vector<int> override = 0;

        void train(std::vector<float>& inputErrors) override;

        [[nodiscard]] auto isValid() const -> errorType override;

        void resetLearningVariables(int batchSize) final;

        auto operator==(const BaseLayer& layer) const -> bool override;
};

template <BaseNeuron N>
template <class Archive>
void Layer<N>::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Layer, BaseLayer>();
    archive& boost::serialization::base_object<BaseLayer>(*this);
    archive& this->numberOfInputs;
    archive& this->neurons;
    archive.template register_type<Dropout>();
    archive.template register_type<L1Regularization>();
    archive.template register_type<L2Regularization>();
    archive.template register_type<ErrorMultiplier>();
    archive.template register_type<Softmax>();
    archive& this->optimizers;
}

}  // namespace snn::internal
#include "Layer.tpp"  // IWYU pragma: keep

extern template class snn::internal::Layer<snn::internal::SimpleNeuron>;
extern template class snn::internal::Layer<snn::internal::RecurrentNeuron>;
extern template class snn::internal::Layer<snn::internal::GatedRecurrentUnit>;
