#pragma once
#include <memory>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include "../Optimizer.hpp"
#include "neuron/BaseNeuron.hpp"

namespace snn::internal
{
    class BaseLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        virtual ~BaseLayer() = default;
        virtual std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const = 0;

        [[nodiscard]] virtual void* getNeuron(int index) = 0;
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

    template <class Archive>
    void BaseLayer::serialize(Archive& ar, unsigned version)
    {
    }
}
