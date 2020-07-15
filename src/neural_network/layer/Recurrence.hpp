#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "SimpleLayer.hpp"
#include "../Optimizer.hpp"
#include "neuron/RecurrentNeuron.hpp"

namespace snn::internal
{
    class Recurrence final : public SimpleLayer<RecurrentNeuron>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        Recurrence() = default;  // use restricted to Boost library only
        Recurrence(LayerModel& model, StochasticGradientDescent* optimizer);
        Recurrence(const Recurrence&) = default;
        ~Recurrence() = default;
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;
    };

    template <class Archive>
    void Recurrence::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Recurrence, SimpleLayer>();
        ar & boost::serialization::base_object<SimpleLayer>(*this);
    }

    inline
    std::vector<float> SimpleLayer<RecurrentNeuron>::output(const std::vector<float>& inputs, bool temporalReset)
    {
        std::vector<float> outputs(this->neurons.size());
        for (int n = 0; n < this->neurons.size(); ++n)
        {
            outputs[n] = neurons[n].output(inputs, temporalReset);
        }
        return outputs;
    }
}
