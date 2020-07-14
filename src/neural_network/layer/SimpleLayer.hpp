#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"

namespace snn::internal
{
    template <class N>
    class SimpleLayer final : public Layer<N>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        SimpleLayer() = default;  // use restricted to Boost library only
        SimpleLayer(LayerModel& model, StochasticGradientDescent* optimizer);
        SimpleLayer(const SimpleLayer&) = default;
        ~SimpleLayer() = default;
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override;
        std::vector<float> backOutput(std::vector<float>& inputErrors) override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class N>
    template <class Archive>
    void SimpleLayer<N>::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<SimpleLayer, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
    }

    #include "SimpleLayer.tpp"
}
