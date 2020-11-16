#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "SimpleLayer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
    class FullyConnected final : public SimpleLayer<SimpleNeuron>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        FullyConnected() = default;  // use restricted to Boost library only
        FullyConnected(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        FullyConnected(const FullyConnected&) = default;
        ~FullyConnected() = default;
        std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;
    };

    template <class Archive>
    void FullyConnected::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<FullyConnected, SimpleLayer>();
        ar & boost::serialization::base_object<SimpleLayer>(*this);
    }
}
