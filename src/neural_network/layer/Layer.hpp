#pragma once
#include <memory>
#include <typeinfo>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "BaseLayer.hpp"
#include "../Optimizer.hpp"
#include "LayerType.hpp"
#include "LayerModel.hpp"

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
        std::vector<float> errors;

    public:
        Layer() = default; // use restricted to Boost library only
        Layer(LayerModel& model, StochasticGradientDescent* optimizer);
        Layer(const Layer&) = default;
        virtual ~Layer() = default;

        // TODO : Can this line be removed ?
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override = 0;

        static const layerType type;
        std::vector<N> neurons;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override = 0;
        std::vector<float> backOutput(std::vector<float>& inputErrors) override = 0;

        [[nodiscard]] Neuron* getNeuron(int index) override final;
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
        ar & this->errors;
        ar & this->neurons;
    }

    #include "Layer.tpp"
}
