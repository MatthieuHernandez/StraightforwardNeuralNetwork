#pragma once
#include "LayerType.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn
{
    struct LayerModel
    {
        layerType type;
        activationFunction activation;
        int numberOfNeurons;
        int numberOfRecurrences;
        int numberOfConvolution;
        int sizeOfConvolutionMatrix;
        int sizeOfInputs[3];
    };
}