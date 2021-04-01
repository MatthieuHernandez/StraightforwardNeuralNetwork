#pragma once
#include <memory>
#include <vector>
#include <boost/serialization/access.hpp>
#include "input/NeuronInput.hpp"
#include "../../optimizer/StochasticGradientDescent.hpp"

template <typename T>
concept Float = std::is_same<float, T>::value;

template <typename T>
concept Int = std::is_same<int, T>::value;

namespace snn::internal
{
    
    template <class N>
    concept HasTemporalOuputMethod =
    requires(N neuron, std::vector<float> inputs)
    {
        {neuron.output(inputs) } -> Float;
    };

    template <class N>
    concept HasNonTemporalOuputMethod =
    requires(N neuron, std::vector<float> inputs)
    {
        {neuron.output(inputs, true) } -> Float;
    };

    template <class N>
    concept HasCommonMethods =
    requires(N neuron)
    {
        {neuron.isValid() } -> Int;
    };

    template <class N>
    concept BaseNeuron2 = HasCommonMethods<N> && (HasTemporalOuputMethod<N> || HasNonTemporalOuputMethod<N>);

    template <class Derived>
    class BaseNeuron
    {
    private:
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        BaseNeuron() = default;
        BaseNeuron(std::shared_ptr<NeuralNetworkOptimizer> optimizer) : optimizer(optimizer) {}
        ~BaseNeuron() = default;
        
        std::shared_ptr<NeuralNetworkOptimizer> optimizer = nullptr;

        template <NeuronInput I>
        [[nodiscard]] float output(const I& inputs) { return static_cast<Derived*>(this)->output22(inputs); }

        template <NeuronInput I>
        [[nodiscard]] float output(const I& inputs, bool reset) { return static_cast<Derived*>(this)->output(inputs, reset); }

        [[nodiscard]] std::vector<float>& backOutput(float error) { return static_cast<Derived*>(this)->backOutput(error); }
        void train(float error) { static_cast<Derived*>(this)->train(error); }

        [[nodiscard]]  int isValid() const { return static_cast<const Derived*>(this)->isValid(); }

        [[nodiscard]] std::vector<float> getWeights() const { return static_cast<const Derived*>(this)->getWeights(); }
        [[nodiscard]] int getNumberOfParameters() const { return static_cast<const Derived*>(this)->getNumberOfParameters(); }
        [[nodiscard]] int getNumberOfInputs() const { return static_cast<const Derived*>(this)->getNumberOfInputs(); }

        bool operator==(const BaseNeuron& neuron) const
        {
            return static_cast<const Derived*>(this)->operator==(static_cast<const Derived&>(neuron));
        }

        bool operator!=(const BaseNeuron& neuron) const
        {
            return !(*this == neuron);
        }
    };

    
    template <class Derived>
    template <class Archive>
    void BaseNeuron<Derived>::serialize(Archive& ar, unsigned version)
    {
        ar.template register_type<StochasticGradientDescent>();
        ar & this->optimizer;
    }
}
