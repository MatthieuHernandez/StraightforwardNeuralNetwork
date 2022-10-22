#pragma once
#include <memory>
#include <typeinfo>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "BaseLayer.hpp"
#include "LayerModel.hpp"
#include "../optimizer/LayerOptimizer.hpp"
#include "../optimizer/LayerOptimizerFactory.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "../optimizer/Dropout.hpp"
#include "../optimizer/L1Regularization.hpp"
#include "../optimizer/L2Regularization.hpp"
#include "../optimizer/ErrorMultiplier.hpp"

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

        [[nodiscard]] virtual Tensor computeOutput(const Tensor& inputs, bool temporalReset) = 0;
        [[nodiscard]] virtual Tensor computeBackOutput(Tensor& inputErrors) = 0;

    public:
        Layer() = default; // use restricted to Boost library only
        Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Layer(const Layer& layer);
        virtual ~Layer() = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override = 0;

        std::vector<N> neurons;
        std::vector<std::unique_ptr<LayerOptimizer>> optimizers;

        Tensor output(const Tensor& inputs, bool temporalReset) override final;
        Tensor outputForTraining(const Tensor& inputs, bool temporalReset) override final;
        Tensor backOutput(Tensor& inputErrors) override final;

        [[nodiscard]] void* getNeuron(int index) override final;
        [[nodiscard]] float getAverageOfAbsNeuronWeights() const override final;
        [[nodiscard]] float getAverageOfSquareNeuronWeights() const override final;
        [[nodiscard]] int getNumberOfInputs() const override final;
        [[nodiscard]] int getNumberOfNeurons() const override final;
        [[nodiscard]] int getNumberOfParameters() const override final;
        [[nodiscard]] std::vector<int> getShapeOfInput() const override = 0;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override = 0;

        void train(Tensor& inputErrors) override final;

        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <BaseNeuron N>
    template <class Archive>
    void Layer<N>::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<Layer, BaseLayer>();
        ar & boost::serialization::base_object<BaseLayer>(*this);
        ar & this->numberOfInputs;
        ar & this->neurons;
        ar.template register_type<Dropout>();
        ar.template register_type<L1Regularization>();
        ar.template register_type<L2Regularization>();
        ar.template register_type<ErrorMultiplier>();
        ar & this->optimizers;
    }

    #include "Layer.tpp"
}
