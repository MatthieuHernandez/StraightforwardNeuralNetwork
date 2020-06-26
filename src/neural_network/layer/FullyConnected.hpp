#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn::internal
{
    class FullyConnected final : public Layer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        FullyConnected() = default;  // use restricted to Boost library only
        FullyConnected(LayerModel& model, StochasticGradientDescent* optimizer);
        FullyConnected(const FullyConnected&) = default;
        ~FullyConnected() = default;
        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override;
        std::vector<float> backOutput(std::vector<float>& inputErrors) override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const FullyConnected& layer) const;
        bool operator!=(const FullyConnected& layer) const;
    };

    template <class Archive>
    void FullyConnected::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<FullyConnected, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
    }
}
