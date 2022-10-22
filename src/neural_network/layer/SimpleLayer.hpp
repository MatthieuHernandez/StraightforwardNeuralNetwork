#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../optimizer/LayerOptimizer.hpp"
#include "neuron/RecurrentNeuron.hpp"
#include "neuron/GatedRecurrentUnit.hpp"

namespace snn::internal
{
    template <BaseNeuron N>
    class SimpleLayer : public Layer<N>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        [[nodiscard]] Tensor computeBackOutput(Tensor& inputErrors) override final;
        [[nodiscard]] Tensor computeOutput(const Tensor& inputs, bool temporalReset) override final;

    public:
        SimpleLayer() = default;  // use restricted to Boost library only
        SimpleLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        SimpleLayer(const SimpleLayer&) = default;
        virtual ~SimpleLayer() = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;

        [[nodiscard]] std::vector<int> getShapeOfInput() const override final;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override final;
        [[nodiscard]] int isValid() const override final;

        bool operator==(const BaseLayer& layer) const override final;
        bool operator!=(const BaseLayer& layer) const override final;
    };

    template <BaseNeuron N>
    template <class Archive>
    void SimpleLayer<N>::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<SimpleLayer<N>, Layer<N>>();
        ar & boost::serialization::base_object<Layer<N>>(*this);
    }

    template<>
    Tensor SimpleLayer<RecurrentNeuron>::computeOutput(const Tensor& inputs, bool temporalReset);

    template<>
    Tensor SimpleLayer<GatedRecurrentUnit>::computeOutput(const Tensor& inputs, bool temporalReset);
    
    #include "SimpleLayer.tpp"
}
