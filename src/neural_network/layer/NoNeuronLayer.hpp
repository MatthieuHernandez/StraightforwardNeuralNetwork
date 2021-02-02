#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include "BaseLayer.hpp"
#include "LayerModel.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "neuron/BaseNeuron.hpp"

namespace snn::internal
{
    class NoNeuronLayer : public BaseLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        int numberOfInputs;
        int numberOfOutputs;

    public:
        NoNeuronLayer() = default; // use restricted to Boost library only
        NoNeuronLayer(LayerModel& model)
        {
            this->numberOfInputs = model.numberOfInputs;
            this->numberOfOutputs = model.numberOfOutputs;
        }
        virtual ~NoNeuronLayer() = default;

        [[nodiscard]] int getNumberOfOutput() const { return numberOfOutputs; }
        [[nodiscard]] void* getNeuron(int index) override;
        [[nodiscard]] int getNumberOfNeurons() const override;
        [[nodiscard]] int getNumberOfParameters() const override;
    };
    
    template <class Archive>
    void NoNeuronLayer::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<NoNeuronLayer, BaseLayer>();
        ar & boost::serialization::base_object<BaseLayer>(*this);
        ar & this->numberOfOutputs;
    }
}
