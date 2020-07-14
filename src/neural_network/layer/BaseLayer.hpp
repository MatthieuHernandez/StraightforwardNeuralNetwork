#pragma once
#include <memory>
#include <boost/serialization/vector.hpp>
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn
{
    struct LayerModel;
}

namespace snn::internal
{
    class BaseLayer
    {
    public:
        virtual std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const = 0;

        [[nodiscard]] virtual Neuron* getNeuron(int index) = 0;
        [[nodiscard]] virtual int getNumberOfInputs() const = 0;
        [[nodiscard]] virtual int getNumberOfNeurons() const = 0;
        [[nodiscard]] virtual int getNumberOfParameters() const = 0;
        [[nodiscard]] virtual std::vector<int> getShapeOfOutput() const = 0;

        virtual std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) = 0;
        virtual std::vector<float> backOutput(std::vector<float>& inputErrors) = 0;
        virtual void train(std::vector<float>& inputErrors) = 0;

        [[nodiscard]] virtual int isValid() const = 0;

        virtual bool operator==(const BaseLayer& layer) const = 0;
        virtual bool operator!=(const BaseLayer& layer) const = 0;
    };
}