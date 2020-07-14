#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "Filter.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn::internal
{
    class Convolution1D final : public FilterLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const override;
        void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const override;

    public:
        Convolution1D() = default; // use restricted to Boost library only
        Convolution1D(LayerModel& model, StochasticGradientDescent* optimizer);
        ~Convolution1D() = default;
        Convolution1D(const Convolution1D&) = default;

        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void Convolution1D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Convolution1D, FilterLayer>();
        ar & boost::serialization::base_object<FilterLayer>(*this);
    }
}
