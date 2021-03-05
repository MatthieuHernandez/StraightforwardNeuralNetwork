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

namespace snn::internal
{
    template <class N>
    class Layer : public BaseLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        int numberOfInputs;

        [[nodiscard]] virtual std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) = 0;
        [[nodiscard]] virtual std::vector<float> computeBackOutput(std::vector<float>& inputErrors) = 0;

    public:
        Layer() = default; // use restricted to Boost library only
        Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        Layer(const Layer& layer);
        virtual ~Layer() = default;

        std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override = 0;

        std::vector<N> neurons;
        std::vector<std::unique_ptr<LayerOptimizer>> optimizers;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override final;
        std::vector<float> outputForTraining(const std::vector<float>& inputs, bool temporalReset) override final;
        std::vector<float> backOutput(std::vector<float>& inputErrors) override final;

        [[nodiscard]] void* getNeuron(int index) override final;
        [[nodiscard]] float getAverageOfAbsNeuronWeights() const override final;
        [[nodiscard]] float getAverageOfSquareNeuronWeights() const override final;
        [[nodiscard]] int getNumberOfInputs() const override final;
        [[nodiscard]] int getNumberOfNeurons() const override final;
        [[nodiscard]] int getNumberOfParameters() const override final;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override = 0;

        void train(std::vector<float>& inputErrors) override final;

        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class N>
    template <class Archive>
    void Layer<N>::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<Layer, BaseLayer>();
        ar & boost::serialization::base_object<BaseLayer>(*this);
        ar & this->numberOfInputs;
        ar & this->neurons;
        ar.template register_type<Dropout>();
        ar & this->optimizers;
    }

    #include "Layer.tpp"
}
