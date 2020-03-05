#pragma once
#include <vector>
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
        std::vector<int> shapeOfInput;
    };
}