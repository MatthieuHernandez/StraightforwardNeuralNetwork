#pragma once
namespace snn
{
    enum class layerOptimizerType
    {
        dropout = 0,
        l1Regularization = 1,
        l2Regularization = 2
    };

    enum class neuralNetworkOptimizerType
    {
        stochasticGradientDescent = 0,
    };
}