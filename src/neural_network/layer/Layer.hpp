#pragma once
#include <memory>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "BaseLayer.hpp"
#include "../Optimizer.hpp"
#include "LayerType.hpp"
#include "LayerModel.hpp"

namespace snn
{
    struct LayerModel;
}

namespace snn::internal
{
    // TODO: use external template to list all layer<N>
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
        ar & this->numberOfInputs;
        ar & this->errors;
        ar & this->neurons;
    }

    //#define TPP_FILE
    //#include "Layer.tpp"
    /*#ifndef TPP_FILE
    #include "Layer.hpp"
    using namespace std;
    using namespace snn;
    using namespace internal;
    #endif*/
}

template <class N>
snn::internal::Layer<N>::Layer(LayerModel& model, StochasticGradientDescent* optimizer)
{
    this->numberOfInputs = model.numberOfInputs;
    this->neurons.reserve(model.numberOfNeurons);
    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(model.numberOfInputsByNeurons, model.activation, optimizer);
    }
}

template <class N>
void snn::internal::Layer<N>::train(std::vector<float>& inputErrors)
{
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        neurons[n].train(inputErrors[n]);
    }
}

template <class N>
int snn::internal::Layer<N>::isValid() const
{
    if (this->neurons.size() != this->getNumberOfNeurons()
        || this->getNumberOfNeurons() < 1
        || this->getNumberOfNeurons() > 1000000)
        return 201;

    int numberOfOutput = 1;
    auto shape = this->getShapeOfOutput();
    for (int s : shape)
        numberOfOutput *= s;

    if (numberOfOutput != this->getNumberOfNeurons())
        return 202;

    for (auto& neuron : this->neurons)
    {
        int err = neuron.isValid();
        if (err != 0)
            return err;
    }
    return 0;
}

template <class N>
snn::internal::Neuron* snn::internal::Layer<N>::getNeuron(int index)
{
    return &this->neurons[index];
}

template <class N>
int snn::internal::Layer<N>::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

template <class N>
int snn::internal::Layer<N>::getNumberOfNeurons() const
{
    return this->neurons.size();
}

template <class N>
int snn::internal::Layer<N>::getNumberOfParameters() const
{
    int sum = 0;
    for (auto& neuron : this->neurons)
    {
        sum += neuron.getNumberOfParameters();
    }
    return sum;
}

template <class N>
bool snn::internal::Layer<N>::operator==(const BaseLayer& layer) const
{
    try
    {
        const Layer& l = dynamic_cast<const Layer&>(layer);
        return typeid(this) == typeid(layer)
            && this->numberOfInputs == l.numberOfInputs
            && this->errors == l.errors
            && this->neurons == l.neurons;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

template <class N>
bool snn::internal::Layer<N>::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}