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

        [[nodiscard]] virtual std::vector<float> computeOutput(const std::vector<float>& inputs,
                                                               bool temporalReset) = 0;
        [[nodiscard]] virtual std::vector<float> computeBackOutput(std::vector<float>& inputErrors) = 0;
        virtual void computeTrain(std::vector<float>& inputErrors) = 0;

    public:
        Layer() = default;  // use restricted to Boost library only
        Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Layer(const Layer& layer);
        virtual ~Layer() = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(
            std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override = 0;

        std::vector<N> neurons;
        std::vector<std::unique_ptr<LayerOptimizer>> optimizers;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) final;
        std::vector<float> outputForTraining(const std::vector<float>& inputs, bool temporalReset) final;
        std::vector<float> backOutput(std::vector<float>& inputErrors) final;

        [[nodiscard]] void* getNeuron(int index) final;
        [[nodiscard]] float getAverageOfAbsNeuronWeights() const final;
        [[nodiscard]] float getAverageOfSquareNeuronWeights() const final;
        [[nodiscard]] int getNumberOfInputs() const final;
        [[nodiscard]] int getNumberOfNeurons() const final;
        [[nodiscard]] int getNumberOfParameters() const final;
        [[nodiscard]] std::vector<int> getShapeOfInput() const override = 0;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override = 0;

        void train(std::vector<float>& inputErrors) final;

        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
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
