#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "FilterLayer.hpp"
#include "../optimizer/StochasticGradientDescent.hpp"
#include "neuron/Neuron.hpp"

namespace snn::internal
{
    class LocallyConnected2D final : public FilterLayer
    {
    private :
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const override;
        void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const override;

    public :
        LocallyConnected2D() = default;  // use restricted to Boost library only
        LocallyConnected2D(LayerModel& model, StochasticGradientDescent* optimizer);
        ~LocallyConnected2D() = default;
        LocallyConnected2D(const LocallyConnected2D&) = default;
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void LocallyConnected2D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<LocallyConnected2D, FilterLayer>();
        ar & boost::serialization::base_object<FilterLayer>(*this);
    }
}
