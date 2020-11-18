#pragma once
#include "OptimizerType.hpp"

namespace snn
{
    struct NeuralNetworkOptimizerModel
    {
        neuralNetworkOptimizerType type;
        float learningRate;
        float momentum;
    };
}
