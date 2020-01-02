#pragma once
#include "LayerType.hpp"
namespace snn::internal
{
    struct LayerModel
    {
        const layerType type;
        const activationFunction activationFunction;
        const int numberOfNeurons;
        const int numberOfRecurrences;
        const int numberOfConvolution;
        const int sizeOfConvolutionMatrix;
        const int sizeOfInputs[3];
    };
}