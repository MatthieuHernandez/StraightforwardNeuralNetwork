#pragma once
#include <array>
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class Convolution2D final : public Layer
    {
    private :
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        static int computeNumberOfInputs(std::array<int, 3> sizeOfInputs); // only use in constructor
        static int computeNumberOfNeurons(int sizeOfConvolutionMatrix, int numberOfConvolution, std::array<int, 3> sizeOfInputs); // only use in constructor
        static int computeNumberOfInputsForNeurones(int sizeOfConvolutionMatrix, std::array<int, 3> sizeOfInputs);

        static std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs);

    public :
        Convolution2D() = default;  // use restricted to Boost library only
        Convolution2D(int numberOfConvolution,
                      int sizeOfConvolutionMatrix,
                      std::array<int, 3> shapeOfInput,
                      activationFunction activation,
                      StochasticGradientDescent* optimizer);
        ~Convolution2D() = default;
        Convolution2D(const Convolution2D&) = default;

        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        int numberOfConvolution;
        int sizeOfConvolutionMatrix;
        std::array<int, 3> shapeofInput;

        std::vector<float> output(const std::vector<float>& inputs) override;
        std::vector<float> backOutput(std::vector<float>& inputsError) override;
        void train(std::vector<float>& inputsError) override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const Convolution2D& layer) const;
        bool operator!=(const Convolution2D& layer) const;
    };

    template <class Archive>
    void Convolution2D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Convolution2D, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
    }
}
