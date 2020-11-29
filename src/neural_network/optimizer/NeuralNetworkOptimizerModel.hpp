#pragma once
#include "OptimizerType.hpp"

namespace snn
{
    struct NeuralNetworkOptimizerModel
    {
        neuralNetworkOptimizerType type;
        float learningRate;
        float momentum;
        float beta1;
        float beta2;
        float epsilon;

    };
}
