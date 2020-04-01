#pragma once
#include <array>
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Convolution.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn {
    struct LayerModel;
}

namespace snn::internal
{
    class Convolution2D final : public Convolution
    {
    private :
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const override;

    public :
        Convolution2D() = default;  // use restricted to Boost library only
        Convolution2D(LayerModel& model, StochasticGradientDescent* optimizer);
        ~Convolution2D() = default;
        Convolution2D(const Convolution2D&) = default;

        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        std::vector<float> output(const std::vector<float>& inputs) override;
        std::vector<float> backOutput(std::vector<float>& inputErrors) override;
        void train(std::vector<float>& inputErrors) override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const Convolution2D& layer) const;
        bool operator!=(const Convolution2D& layer) const;
    };

    template <class Archive>
    void Convolution2D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Convolution2D, Convolution>();
        ar & boost::serialization::base_object<Convolution>(*this);
    }
}
