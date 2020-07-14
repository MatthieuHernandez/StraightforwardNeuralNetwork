#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Filter.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn {
    struct LayerModel;
}

namespace snn::internal
{
    class Convolution2D final : public Filter
    {
    private :
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const override;
        void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const override;

    public :
        Convolution2D() = default;  // use restricted to Boost library only
        Convolution2D(LayerModel& model, StochasticGradientDescent* optimizer);
        ~Convolution2D() = default;
        Convolution2D(const Convolution2D&) = default;
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void Convolution2D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Convolution2D, Filter>();
        ar & boost::serialization::base_object<Filter>(*this);
    }
}
