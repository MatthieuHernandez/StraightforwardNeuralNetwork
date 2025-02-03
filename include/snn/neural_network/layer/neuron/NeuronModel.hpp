#pragma once
#include "activation_function/ActivationFunction.hpp"

namespace snn
{
struct NeuronModel
{
        int numberOfInputs = -1;
        int batchSize = -1;
        int numberOfWeights = -1;
        float bias = 1.0F;
        activation activationFunction;
};
}  // namespace snn
