#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <memory>
#include <typeinfo>

#include "../optimizer/Dropout.hpp"
#include "../optimizer/ErrorMultiplier.hpp"
#include "../optimizer/L1Regularization.hpp"
#include "../optimizer/L2Regularization.hpp"
#include "../optimizer/LayerOptimizer.hpp"
#include "../optimizer/LayerOptimizerFactory.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "../optimizer/Softmax.hpp"
#include "BaseLayer.hpp"
#include "LayerModel.hpp"

namespace snn::internal
{
template <BaseNeuron N>
class Layer : public BaseLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        int numberOfInputs;

        [[nodiscard]] virtual auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> = 0;
        [[nodiscard]] virtual auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> = 0;
        virtual void computeTrain(std::vector<float>& inputErrors) = 0;

    public:
        Layer() = default;  // use restricted to Boost library only
        Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Layer(const Layer& layer);
        virtual ~Layer() = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> override = 0;

        std::vector<N> neurons;
        std::vector<std::unique_ptr<LayerOptimizer>> optimizers;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) final;
        std::vector<float> outputForTraining(const std::vector<float>& inputs, bool temporalReset) final;
        std::vector<float> backOutput(std::vector<float>& inputErrors) final;

        [[nodiscard]] auto getNeuron(int index) -> void* final;
        [[nodiscard]] auto getAverageOfAbsNeuronWeights() const -> float final;
        [[nodiscard]] auto getAverageOfSquareNeuronWeights() const -> float final;
        [[nodiscard]] auto getNumberOfInputs() const -> int final;
        [[nodiscard]] auto getNumberOfNeurons() const -> int final;
        [[nodiscard]] auto getNumberOfParameters() const -> int final;
        [[nodiscard]] auto getShapeOfInput() const -> std::vector<int> override = 0;
        [[nodiscard]] auto getShapeOfOutput() const -> std::vector<int> override = 0;

        void train(std::vector<float>& inputErrors) final;

        [[nodiscard]] auto isValid() const -> ErrorType override;

        auto operator==(const BaseLayer& layer) const -> bool override;
        auto operator!=(const BaseLayer& layer) const -> bool override;
};

template <BaseNeuron N>
template <class Archive>
void Layer<N>::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<Layer, BaseLayer>();
    ar& boost::serialization::base_object<BaseLayer>(*this);
    ar& this->numberOfInputs;
    ar& this->neurons;
    ar.template register_type<Dropout>();
    ar.template register_type<L1Regularization>();
    ar.template register_type<L2Regularization>();
    ar.template register_type<ErrorMultiplier>();
    ar.template register_type<Softmax>();
    ar& this->optimizers;
}

#include "Layer.tpp"
}  // namespace snn::internal
