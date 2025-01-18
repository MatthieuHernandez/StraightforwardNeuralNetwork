#pragma once
namespace snn
{
enum class layerOptimizerType
{
    dropout = 0,
    l1Regularization = 1,
    l2Regularization = 2,
    errorMultiplier = 3,
    softmax = 4
};

enum class neuralNetworkOptimizerType
{
    stochasticGradientDescent = 0,
};
}  // namespace snn