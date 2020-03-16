#pragma once
#include <array>
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn {
    struct LayerModel;
}

namespace snn::internal
{
    class Convolution1D final : public Layer
    {
    private :
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        static std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs);

    public :
        Convolution1D() = default;  // use restricted to Boost library only
        Convolution1D(LayerModel& model, StochasticGradientDescent* optimizer);
        ~Convolution1D() = default;
        Convolution1D(const Convolution1D&) = default;

        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        int numberOfConvolution;
        int sizeOfConvolutionMatrix;
        std::vector<int> shapeofInput;

        std::vector<float> output(const std::vector<float>& inputs) override;
        std::vector<float> backOutput(std::vector<float>& inputsError) override;
        void train(std::vector<float>& inputsError) override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const Convolution1D& layer) const;
        bool operator!=(const Convolution1D& layer) const;
    };

    template <class Archive>
    void Convolution1D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Convolution1D, Convolution>();
        ar & boost::serialization::base_object<Convolution>(*this);
    }
}
