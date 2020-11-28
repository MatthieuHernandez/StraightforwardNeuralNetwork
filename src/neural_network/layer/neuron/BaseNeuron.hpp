#pragma once
#include <memory>
#include <vector>
#include <boost/serialization/access.hpp>
#include "../../optimizer/StochasticGradientDescent.hpp"
#include "../../optimizer/Adam.hpp"

namespace snn::internal
{
    class NeuralNetworkOptimizer;

    template <class Derived>
    class BaseNeuron
    {
    private:
        friend class StochasticGradientDescent;
        friend class Adam;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        BaseNeuron() = default;
        BaseNeuron(std::shared_ptr<NeuralNetworkOptimizer> optimizer) : optimizer(optimizer) {}
        ~BaseNeuron() = default;
        
        std::shared_ptr<NeuralNetworkOptimizer> optimizer = nullptr;

        [[nodiscard]] float output(const std::vector<float>& inputs) { return static_cast<Derived*>(this)->output(inputs); }
        [[nodiscard]] float output(const std::vector<float>& inputs, bool reset) { return static_cast<Derived*>(this)->output(inputs, reset); }

        [[nodiscard]]  std::vector<float>& backOutput(float error) { return static_cast<Derived*>(this)->backOutput(error); }
         void train(float error) { static_cast<Derived*>(this)->train(error); }

        [[nodiscard]]  int isValid() const { return static_cast<Derived*>(this)->isValid(); }

        [[nodiscard]] std::vector<float> getWeights() { return static_cast<Derived*>(this)->getWeights(); }
        [[nodiscard]] int getNumberOfParameters() { return static_cast<Derived*>(this)->getNumberOfParameters(); }
        [[nodiscard]] int getNumberOfInputs() { return static_cast<Derived*>(this)->getNumberOfInputs(); }

        bool operator==(const BaseNeuron& neuron) const
        {
            return static_cast<Derived*>(this) == static_cast<Derived*>(neuron);
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
