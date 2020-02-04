#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class Convolution2D : public Layer
    {
    private :
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        static int computeNumberOfInputs(int sizeOfInputs[3]); // only use in constructor
        static int computeNumberOfNeurons(int numberOfConvolution, int sizeOfInputs[3]); // only use in constructor
        static int computeNumberOfInputsForNeurones(int sizeOfConvolutionMatrix, int sizeOfInputs[3]);

    public :
        Convolution2D() = default;  // use restricted to Boost library only
        Convolution2D(int numberOfConvolution,
                      int sizeOfConvolutionMatrix,
                      int sizeOfInputs[3],
                      activationFunction activation,
                      StochasticGradientDescent* optimizer);
        ~Convolution2D() = default;
        Convolution2D(const Convolution2D&) = default;

        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        int numberOfConvolution;
        int sizeOfConvolutionMatrix;
        int sizeOfInputs[3];

        std::vector<float> output(const std::vector<float>& inputs) override;
        std::vector<float> backOutput(std::vector<float>& inputsError) override;
        void train(std::vector<float>& inputsError) override;

        int isValid() const override;

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
