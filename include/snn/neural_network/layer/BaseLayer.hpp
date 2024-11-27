#pragma once
#include <memory>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "neuron/BaseNeuron.hpp"

namespace snn::internal
{
    class BaseLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize([[maybe_unused]] Archive& ar, [[maybe_unused]] const unsigned version) {}

    public:
        virtual ~BaseLayer() = default;
        virtual std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const = 0;

        [[nodiscard]] virtual void* getNeuron(int index) = 0;
        [[nodiscard]] virtual float getAverageOfAbsNeuronWeights() const = 0;
        [[nodiscard]] virtual float getAverageOfSquareNeuronWeights() const = 0;
        [[nodiscard]] virtual int getNumberOfInputs() const = 0;
        [[nodiscard]] virtual int getNumberOfNeurons() const = 0;
        [[nodiscard]] virtual int getNumberOfParameters() const = 0;
        [[nodiscard]] virtual std::vector<int> getShapeOfInput() const = 0;
        [[nodiscard]] virtual std::vector<int> getShapeOfOutput() const = 0;

        [[nodiscard]] virtual std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) = 0;
        [[nodiscard]] virtual std::vector<float> outputForTraining(const std::vector<float>& inputs, bool temporalReset) = 0;
        [[nodiscard]] virtual std::vector<float> backOutput(std::vector<float>& inputErrors) = 0;
        virtual void train(std::vector<float>& inputErrors) = 0;

        [[nodiscard]] virtual int isValid() const = 0;

        [[nodiscard]] virtual std::string summary() const = 0;

        virtual bool operator==(const BaseLayer& layer) const = 0;
        virtual bool operator!=(const BaseLayer& layer) const = 0;
    };
}
