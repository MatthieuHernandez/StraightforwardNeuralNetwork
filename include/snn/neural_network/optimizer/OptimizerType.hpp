#pragma once
#include <cstdint>

namespace snn
{
enum class layerOptimizerType : uint8_t
{
    dropout = 0,
    l1Regularization = 1,
    l2Regularization = 2,
    errorMultiplier = 3,
    softmax = 4
};

enum class neuralNetworkOptimizerType : uint8_t
{
    stochasticGradientDescent = 0,
};
}  // namespace snn