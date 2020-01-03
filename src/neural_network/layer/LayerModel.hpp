#pragma once
#include "LayerType.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    struct LayerModel
    {
        const layerType type;
        const activationFunction activation;
        const int numberOfInputs;
        const int numberOfNeurons;
        const int numberOfRecurrences;
        const int numberOfConvolution;
        const int sizeOfConvolutionMatrix;
        const int sizeOfInputs[3];
    };
}