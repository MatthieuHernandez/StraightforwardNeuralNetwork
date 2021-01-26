#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include "BaseLayer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "neuron/BaseNeuron.hpp"

namespace snn::internal
{
    class NoNeuronLayer : public BaseLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version) {}

    public:
        virtual ~NoNeuronLayer() = default;

        [[nodiscard]] void* getNeuron(int index) override;
        [[nodiscard]] int getNumberOfNeurons() const override;
        [[nodiscard]] int getNumberOfParameters() const override;
    };
}
